mod types;
mod arraybuilder;

use std::collections::HashMap;
use std::thread;
use std::time::Instant;

use anyhow::{anyhow};
use arrow::array::{Array, ArrayRef as ArrowArrayRef};
use mimalloc::MiMalloc;
use pyo3::prelude::*;
use pyo3_arrow::error::PyArrowResult;
use pyo3_arrow::{PyArray, PyChunkedArray};
use sqlparser::dialect;
use sqlparser::ast::{Insert, SetExpr, Statement, Values};

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

/// A Python module implemented in Rust.
#[pymodule]
fn sql2arrow(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(load_from_local, m)?)?;
    Ok(())
}

#[pyfunction]
fn load_from_local(file_paths: Vec<String>, columns: Vec<Vec<String>>) ->  PyArrowResult<Vec<Vec<PyArray>>> {
    if file_paths.is_empty() {
        return Err(pyo3_arrow::error::PyArrowError::PyErr(
            pyo3::exceptions::PyRuntimeError::new_err("file_paths is empty")
        ));
    }

    if file_paths.len() > 32 {
        return Err(pyo3_arrow::error::PyArrowError::PyErr(
            pyo3::exceptions::PyRuntimeError::new_err("load too many files at once. maximum: 32")
        ));
    }
    
    let (tx, rx) = crossbeam_channel::bounded::<anyhow::Result<(usize, Vec<ArrowArrayRef>)>>(file_paths.len());

    let mut handlers = Vec::with_capacity(file_paths.len());
    let mut i : usize = 0;
    for file_path in &file_paths {
        let tx_thread = tx.clone();
        let columns_thread = columns.clone();
        let file_path_thread = file_path.clone();
        let i_thread = i;
        i += 1;

        let handler = thread::spawn(move || {
            match load_insert_sql_to_arrref(&file_path_thread, columns_thread, i_thread) {
                Ok(arr_refs) => {
                    let _ = tx_thread.send(Ok((i_thread, arr_refs)));
                },
                Err(e) => {
                    let _ = tx_thread.send(Err(e));
                }
            }
            drop(tx_thread)
        });



        handlers.push(handler);
    }
    drop(tx);

    let mut ret_pyarrs = Vec::<Vec<PyArray>>::with_capacity(file_paths.len());
    for _ in 0..file_paths.len() {
        ret_pyarrs.push(Vec::<PyArray>::with_capacity(columns.len()));
    }

    let mut res = anyhow::Ok(());
    for array_refs_res in rx {
        match array_refs_res {
            Ok((i, arr_refs)) => {
                let pyarrs = ret_pyarrs.get_mut(i).unwrap();
                for arr_ref in arr_refs {
                    pyarrs.push(PyArray::from_array_ref(arr_ref));
                }
            },
            Err(e) => {
                res = Err(e);
                break;
            }
        }
    }

    for handler in handlers {
        let _ = handler.join();
    }
    
    if res.is_err() {
        return Err(pyo3_arrow::error::PyArrowError::PyErr(
            pyo3::exceptions::PyRuntimeError::new_err(res.err().unwrap().to_string())
        ));
    }
    

    return Ok(ret_pyarrs);

    
}


/**
 * columns
 * [
 *     index => [column_name,  data_type]
 * ]
 */
fn load_insert_sql_to_arrref(file_path: &str, columns : Vec<Vec<String>>, idx_thread : usize) -> anyhow::Result<Vec<ArrowArrayRef>> {
    use std::env;
    let mut is_debug = false;
    match env::var("SQL2ARROW_DEBUG") {
        Ok(value) => is_debug = true,
        _ => {}
    }

    if file_path.is_empty() || columns.is_empty() {
        return Err(anyhow!("file_path is empty or columns is empty"));
    }

    let mut dt_vec = Vec::<&str>::with_capacity(columns.len());
    let mut column_name_to_outidx = HashMap::<String, usize>::with_capacity(columns.len());
    let mut i : usize = 0;
    for v in &columns {
        if v.is_empty() || v.len() != 2 {
            return Err(anyhow!("invalid columns"));
        }
        dt_vec.push(&v[1]);
        column_name_to_outidx.insert(v[0].clone(), i);
        i += 1;
    }

    let buffer_load_start_time = Instant::now();
    let row_schema : types::RowSchema = dt_vec.try_into()?;
    let buffer = std::fs::read_to_string(file_path)?;
    if is_debug {
        print!("thread idx: {} load buffer. {:?}\n", idx_thread, buffer_load_start_time.elapsed());
    }


    let parse_sql_start_time = Instant::now();
    let dialect = dialect::MySqlDialect{};
    let ast = sqlparser::parser::Parser::parse_sql(&dialect, &buffer)?;
    drop(buffer);
    if is_debug {
        print!("thread idx: {} parse sql. {:?}\n", idx_thread, parse_sql_start_time.elapsed());
    }
    
    //caculate the builder capacity
    let mut capacity: usize = 0;
    let mut val_idx_to_outidx = HashMap::<usize, usize>::with_capacity(columns.len());

    let build_arr_start_time = Instant::now();
    for st in &ast {
        match st {
            Statement::Insert(Insert{source, columns, ..}) => {
                if !columns.is_empty() {
                    //match the column names
                    let mut val_idx = 0;
                    for col in columns {
                        if column_name_to_outidx.contains_key(col.value.as_str()) {
                            val_idx_to_outidx.insert(val_idx, column_name_to_outidx.get(col.value.as_str()).unwrap().clone());
                            column_name_to_outidx.remove(col.value.as_str());
                        }
                        val_idx += 1;
                    }

                    if !column_name_to_outidx.is_empty() {
                        let not_exists_columns_name : Vec<String> = column_name_to_outidx.keys().cloned().collect();
                        return Err(anyhow!(format!("these columns: {} not exists", not_exists_columns_name.join(","))));
                    }
                } else {
                    //Insert Into xxx VALUES(xxx,xxx)
                    //no columns
                    for (_, outidx) in column_name_to_outidx.iter() {
                        val_idx_to_outidx.insert(outidx.clone(), outidx.clone());
                    }
                }

                match source.as_ref().unwrap().body.as_ref() {
                    SetExpr::Values(Values{  rows, .. }) => {
                        capacity += rows.len();
                    },
                    _ => (),
                };
            },
            _ => (),
        };
        
    }


    let mut builders = row_schema.create_row_array_builders(capacity);

    for st in ast {
        match st {
            Statement::Insert(Insert{source, ..}) => {
                match source.as_ref().unwrap().body.as_ref() {
                    SetExpr::Values(Values{  rows, .. }) => {
                        for row in rows {
                            for (val_idx, outidx) in val_idx_to_outidx.iter() {
                                let b = builders.get_mut(outidx.clone()).unwrap();
                                let dt = row_schema.get(outidx.clone()).unwrap();
                                let expr = row.get(val_idx.clone()).unwrap();
                                
                                arraybuilder::append_value_to_builder(b, dt, expr)?;
                            }
                        }
                    },
                    _ => (),
                };
            },
            _ => (),
        };
    }
    
    if is_debug {
        print!("thread idx: {} build arr. {:?}\n", idx_thread, build_arr_start_time.elapsed());
    }

    let builder_finish_start_time = Instant::now();
    let mut arrays = Vec::<ArrowArrayRef>::with_capacity(builders.len());
    
    for mut b in builders {
        let arr_ref = b.finish();
        arrays.push(arr_ref);
    }

    if is_debug {
        print!("thread idx: {} builder finish. {:?}\n", idx_thread, builder_finish_start_time.elapsed());
    }

    Ok(arrays)
}