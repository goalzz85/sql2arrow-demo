mod types;
mod arraybuilder;

use std::collections::HashMap;

use anyhow::anyhow;
use arrow::array::ArrayRef as ArrowArraryRef;
use pyo3::prelude::*;
use pyo3_arrow::error::PyArrowResult;
use pyo3_arrow::PyArray;
use sqlparser::dialect;
use sqlparser::ast::{Insert, SetExpr, Statement, Values};

/// A Python module implemented in Rust.
#[pymodule]
fn sql2arrow(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(load_from_local, m)?)?;
    Ok(())
}

#[pyfunction]
fn load_from_local(file_path: &str, columns: Vec<Vec<String>>) ->  PyArrowResult<Vec<PyArray>> {
    match load_insert_sql_to_df(file_path, columns) {
        Ok(arr_refs) => {
            let mut res = Vec::<PyArray>::with_capacity(arr_refs.len());
            for arr_ref in arr_refs {
                res.push(PyArray::from_array_ref(arr_ref));
            }
            return Ok(res);
        },
        Err(e) => {
            return Err(pyo3_arrow::error::PyArrowError::PyErr(
                pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
            ));
        }
    }
}


/**
 * columns
 * [
 *     index => [column_name,  data_type]
 * ]
 */
fn load_insert_sql_to_df(file_path: &str, columns : Vec<Vec<String>>) -> anyhow::Result<Vec<ArrowArraryRef>> {
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

    let row_schema : types::RowSchema = dt_vec.try_into()?;
    let buffer = std::fs::read_to_string(file_path)?;

    let dialect = dialect::MySqlDialect{};
    let ast = sqlparser::parser::Parser::parse_sql(&dialect, &buffer)?;
    //caculate the builder capacity
    let mut capacity: usize = 0;
    let mut val_idx_to_outidx = HashMap::<usize, usize>::with_capacity(columns.len());

    for st in &ast {
        match st {
            Statement::Insert(Insert{source, columns, ..}) => {
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
    
    let mut arrays = Vec::<ArrowArraryRef>::with_capacity(builders.len());
    
    for mut b in builders {
        let arr_ref = b.finish();
        arrays.push(arr_ref);
    }

    Ok(arrays)
}