mod types;
mod arraybuilder;
mod partition;


use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::sync::Arc;
use std::thread;
use std::time::Instant;

use flate2::read::GzDecoder;

use anyhow::{anyhow};
use arrow::array::{Array, ArrayRef as ArrowArrayRef};
use arrow::compute::{SortColumn, TakeOptions};
use arrow_array::UInt32Array;
use mimalloc::MiMalloc;
use partition::{get_parition_key_from_first_val, parse_partition_func, DefaultPartition, PartitionFunc, PartitionKey};
use pyo3::{iter, prelude::*};
use pyo3_arrow::error::PyArrowResult;
use pyo3_arrow::{PyArray, PyChunkedArray};
use sqlparser::dialect;
use sqlparser::ast::{Insert, SetExpr, Statement, Values};
use types::ColumnArrStrDef;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

/// A Python module implemented in Rust.
#[pymodule]
fn sql2arrow(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(load_from_local, m)?)?;
    Ok(())
}

#[pyfunction]
#[pyo3(signature = (file_paths, columns, is_gzip=false, partition_type=None, partition_confs=None))]
fn load_from_local(file_paths: Vec<String>, columns: ColumnArrStrDef, is_gzip : bool, partition_type : Option<&str>, partition_confs : Option<HashMap<String, String>>) ->  PyArrowResult<Vec<Vec<PyArray>>> {
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

    let mut is_have_partition = false;
    let mut partition_func : Arc<dyn PartitionFunc> = Arc::new(DefaultPartition{});
    if let Some(partition_type)  = partition_type {
        if let Some(partition_confs) = partition_confs {
            match parse_partition_func(partition_type, partition_confs, &columns) {
                Ok(pf) => {
                    partition_func = pf.into();
                    is_have_partition = true;
                },
                Err(e) => {
                    return Err(pyo3_arrow::error::PyArrowError::PyErr(
                        pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
                    ));
                }
            }
        }
    }
    
    
    let res = if !is_have_partition {
        match load_from_local_without_partition(file_paths, columns, is_gzip) {
            Ok(pyarrs) => PyArrowResult::Ok(pyarrs),
            Err(e) => Err(pyo3_arrow::error::PyArrowError::PyErr(
                    pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
                )),
        }
    } else {
        match load_with_partition_from_local(file_paths, columns, partition_func, is_gzip) {
            Ok(pyarrs) => PyArrowResult::Ok(pyarrs),
            Err(e) => Err(pyo3_arrow::error::PyArrowError::PyErr(
                pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
            )),
        }
    };

    res
}


fn load_from_local_without_partition(file_paths: Vec<String>, columns: ColumnArrStrDef, is_gzip : bool) -> anyhow::Result<Vec<Vec<PyArray>>> {
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
            match load_insert_sql_to_arrref(&file_path_thread, columns_thread, is_gzip, i_thread) {
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
        return Err(res.err().unwrap());
    }
    

    return Ok(ret_pyarrs);
}


fn load_with_partition_from_local(file_paths: Vec<String>, columns: ColumnArrStrDef,  partition_func : Arc<dyn PartitionFunc>, is_gzip : bool) -> anyhow::Result<Vec<Vec<PyArray>>> {
    fn get_sorted_indices_from_multi_cols(arr_refs : &Vec<ArrowArrayRef>) -> anyhow::Result<UInt32Array> {
        let mut sort_cols = Vec::<SortColumn>::with_capacity(arr_refs.len());
        for arr_ref in arr_refs {
            let sort_col = SortColumn {
                values : arr_ref.clone(),
                options: None,
            };
            sort_cols.push(sort_col);
        }

        return Ok(arrow::compute::lexsort_to_indices(&sort_cols, None)?);
    }

    fn data_to_partitioned_arr_refs(arr_refs: &Vec<ArrowArrayRef>, partition_val_arr_refs: &Vec<ArrowArrayRef>, sorted_indices : &UInt32Array) -> anyhow::Result<HashMap<PartitionKey, Vec<ArrowArrayRef>>> {
        let take_opt = TakeOptions{check_bounds:true};
        let sorted_arr_refs = arrow::compute::take_arrays(&arr_refs, &sorted_indices, Some(take_opt.clone()))?;
        let sorted_partition_val_arr_refs = arrow::compute::take_arrays(&partition_val_arr_refs, &sorted_indices, Some(take_opt.clone()))?;
        let partitions = arrow::compute::partition(&sorted_partition_val_arr_refs)?;

        let mut res = HashMap::<PartitionKey, Vec<ArrowArrayRef>>::with_capacity(partitions.len());

        for (_, r) in partitions.ranges().iter().enumerate() {
            

            let mut partitioned_arr_refs = Vec::<ArrowArrayRef>::with_capacity(sorted_arr_refs.len());
            for arr_ref in &sorted_arr_refs {
                let partitioned_arr_ref = arr_ref.slice(r.start, r.end - r.start);
                partitioned_arr_refs.push(partitioned_arr_ref);
            }

            let mut partitioned_val_arr_refs = Vec::<ArrowArrayRef>::with_capacity(sorted_partition_val_arr_refs.len());
            for arr_ref in &sorted_partition_val_arr_refs {
                let partitioned_val_arr_ref = arr_ref.slice(r.start, r.end - r.start);
                partitioned_val_arr_refs.push(partitioned_val_arr_ref);
            }

            let partition_key = get_parition_key_from_first_val(&partitioned_val_arr_refs)?;

            res.insert(partition_key, partitioned_arr_refs);
        }

        

        return Ok(res);
    }


    let (tx, rx) = crossbeam_channel::bounded::<anyhow::Result<HashMap<PartitionKey, Vec<ArrowArrayRef>>>>(file_paths.len());

    let mut handlers = Vec::with_capacity(file_paths.len());
    let mut i : usize = 0;
    for file_path in &file_paths {
        let tx_thread = tx.clone();
        let columns_thread = columns.clone();
        let file_path_thread = file_path.clone();
        let partition_func_thread = partition_func.clone();
        let i_thread = i;
        i += 1;

        let handler = thread::spawn(move || {

            let res_for_send : anyhow::Result<HashMap<PartitionKey, Vec<ArrowArrayRef>>> = (move || {
                let arr_refs = load_insert_sql_to_arrref(&file_path_thread, columns_thread, is_gzip, i_thread)?;

                let mut is_debug = false;
                match std::env::var("SQL2ARROW_DEBUG") {
                    Ok(value) => is_debug = true,
                    _ => {}
                }
                let partition_start_time = Instant::now();
                let partition_val_arr_refs = partition_func_thread.transform(&arr_refs)?;
                let indices = get_sorted_indices_from_multi_cols(&partition_val_arr_refs)?;
                let ret = data_to_partitioned_arr_refs(&arr_refs, &partition_val_arr_refs, &indices);
                if is_debug {
                    print!("thread idx: {} partition. {:?}\n", i_thread, partition_start_time.elapsed());
                }
                ret
            })();

            match res_for_send {
                Ok(hash_arr_refs) => {
                    let _ = tx_thread.send(Ok(hash_arr_refs));
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
    
    let mut hash_arr_refs_batch = HashMap::<PartitionKey, Vec<Vec<ArrowArrayRef>>>::new();

    let mut res = anyhow::Ok(());
    for array_refs_res in rx {
        match array_refs_res {
            Ok(hash_arr_refs) => {
                for (partition_key, arr_refs) in hash_arr_refs {
                    if !hash_arr_refs_batch.contains_key(&partition_key) {
                        let arr_refs_batch = Vec::<Vec<ArrowArrayRef>>::with_capacity(file_paths.len());
                        hash_arr_refs_batch.insert(partition_key.clone(), arr_refs_batch);
                    }
                    let arr_refs_batch = hash_arr_refs_batch.get_mut(&partition_key).unwrap();
                    arr_refs_batch.push(arr_refs);
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
        return Err(res.err().unwrap());
    }

    let mut is_debug = false;
    match std::env::var("SQL2ARROW_DEBUG") {
        Ok(value) => is_debug = true,
        _ => {}
    }

    let combine_partition_start_time = Instant::now();

    let mut ret_pyarrs = Vec::<Vec<PyArray>>::with_capacity(hash_arr_refs_batch.len());
    for (_, arr_refs_batch) in &hash_arr_refs_batch {
        let mut vertical_arr_refs = vec![Vec::<ArrowArrayRef>::with_capacity(hash_arr_refs_batch.len()); columns.len()];
        for arr_refs in arr_refs_batch {
            for (idx, arr_ref) in arr_refs.iter().enumerate() {
                vertical_arr_refs.get_mut(idx).unwrap().push(arr_ref.clone());
            }
        }

        let mut new_arr_refs = Vec::<PyArray>::with_capacity(columns.len());
        for col_arr_refs in vertical_arr_refs {
            let arr_refs_for_concat : Vec<&dyn Array> = col_arr_refs.iter().map(|arc| arc.as_ref()).collect();
            let arr_ref = arrow::compute::concat(arr_refs_for_concat.as_slice())?;
            new_arr_refs.push(PyArray::from_array_ref(arr_ref));
        }

        ret_pyarrs.push(new_arr_refs);
    }
    
    if is_debug {
        print!("combine partition. {:?}\n", combine_partition_start_time.elapsed());
    }

    return Ok(ret_pyarrs);
}
/**
 * columns
 * [
 *     index => (column_name,  data_type)
 * ]
 */
fn load_insert_sql_to_arrref(file_path: &str, columns : ColumnArrStrDef, is_gzip : bool, idx_thread : usize) -> anyhow::Result<Vec<ArrowArrayRef>> {
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
        dt_vec.push(&v.1);
        column_name_to_outidx.insert(v.0.clone(), i);
        i += 1;
    }

    let buffer_load_start_time = Instant::now();
    let row_schema : types::RowSchema = dt_vec.try_into()?;
    let buffer = if is_gzip {
        let file = File::open(file_path)?;
        let mut decoder = GzDecoder::new(file);
        let mut buffer = String::new();
        decoder.read_to_string(&mut buffer)?;

        buffer
    } else {
        std::fs::read_to_string(file_path)?
    };

    if is_debug {
        print!("thread idx: {} load buffer. {:?}\n", idx_thread, buffer_load_start_time.elapsed());
    }


    let dialect = dialect::MySqlDialect{};
    let mut sql_parser = sqlparser::parser::Parser::new(&dialect);
    sql_parser = sql_parser.try_with_sql(&buffer)?;
    
    let mut val_idx_to_outidx = HashMap::<usize, usize>::with_capacity(columns.len());

    let mut expecting_statement_delimiter = false;

    let mut builders = row_schema.create_row_array_builders(10000);
    //loop statement
    loop {
        while sql_parser.consume_token(&sqlparser::tokenizer::Token::SemiColon) {
            expecting_statement_delimiter = false;
        }

        match sql_parser.peek_token().token {
            sqlparser::tokenizer::Token::EOF => break,

            // end of statement
            sqlparser::tokenizer::Token::Word(word) => {
                if expecting_statement_delimiter && word.keyword == sqlparser::keywords::Keyword::END {
                    break;
                }
            }
            _ => {}
        }

        if expecting_statement_delimiter {
            return sql_parser.expected("end of statement", sql_parser.peek_token())?;
        }

        let statement = sql_parser.parse_statement()?;
        if val_idx_to_outidx.is_empty() {
            match &statement {
                Statement::Insert(Insert{columns, ..}) => {
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
                },
                _ => (),
            }
        }

        
        match statement {
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
        }
    } //end of loop

    let mut arrays = Vec::<ArrowArrayRef>::with_capacity(builders.len());
    for mut b in builders {
        let arr_ref = b.finish();
        arrays.push(arr_ref);
    }

    Ok(arrays)
}