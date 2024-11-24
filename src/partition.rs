use std::{borrow::BorrowMut, collections::HashMap, hash::Hash};
use arrow_array::ArrayRef;
use arrow_schema::{DataType, TimeUnit};
use iceberg::{spec::Transform, transform::{self, create_transform_function, BoxedTransformFunction}};
use pyo3::types::PyDict;
use crate::{partition, types::ColumnArrStrDef};
use anyhow::{anyhow};



#[derive(Debug, PartialEq, Eq, Clone, Hash)]
pub(crate) enum PartitionType {
    IceBerg,
}

pub trait PartitionFunc: Send + Sync {
    fn transform(&self, arr_refs : &Vec<ArrayRef>) -> anyhow::Result<Vec<ArrayRef>>;
}

pub type PartitionKey = Vec<u8>;


pub fn parse_partition_func(partition_type : &str, conf : HashMap<String, String>, columns_def : &ColumnArrStrDef) -> anyhow::Result<Box<dyn PartitionFunc>> {
    let partition_func = match partition_type {
        "iceberg" => {
            let mut col_partition_defs = Vec::with_capacity(conf.len());
            for (col, transform) in conf.iter() {
                col_partition_defs.push((col.as_ref(), transform.as_ref()));
            }

            Box::new(IceBergPartition::from(&col_partition_defs, columns_def)?)
        },
        _ => {
            return Err(anyhow!("not support partition type {:?}, 'iceberg' is the valid partition type", partition_type));
        }
    };

    return Ok(partition_func);
}

pub struct DefaultPartition {

}
impl PartitionFunc for DefaultPartition {
    fn transform(&self, arr_refs : &Vec<ArrayRef>) -> anyhow::Result<Vec<ArrayRef>> {
        Err(anyhow!("not implement the PartitionFunc"))
    }
}


pub struct IceBergPartition {
    col_idxs : Vec<usize>,
    transforms: Vec<Transform>
}

impl IceBergPartition {

    /**
     * col_partition_defs:
     * [
     *     ("column name", "parition transform string")
     * ]
     */
    fn from(col_partition_defs : &Vec<(&str, &str)>, columns_def : &ColumnArrStrDef) -> anyhow::Result<Self> {
        if col_partition_defs.is_empty() {
            return Err(anyhow!("partition transforms is empty"));
        }

        let mut col_idxs = Vec::<usize>::with_capacity(col_partition_defs.len());
        let mut col_transforms = Vec::<Transform>::with_capacity(col_partition_defs.len());

        for (col_name, transform_str) in col_partition_defs {
            let mut is_have_col = false;
            for (idx, (col_name_def, col_data_type_def)) in columns_def.iter().enumerate() {
                if col_name.eq(col_name_def) {
                    is_have_col = true;
                    col_idxs.push(idx);
                }
            }
            if !is_have_col {
                return Err(anyhow!("not found column name {:?}", col_name));
            }

            //get iceberg transform
            let tf = transform_str.parse()?;
            col_transforms.push(tf);
        }

        return Ok(IceBergPartition{
            col_idxs: col_idxs,
            transforms: col_transforms,
        });
    }
}


impl PartitionFunc for IceBergPartition {
    fn transform(&self, arr_refs : &Vec<ArrayRef>) -> anyhow::Result<Vec<ArrayRef>> {
        let mut res_arr_refs = Vec::<ArrayRef>::with_capacity(arr_refs.len());

        for (i, col_idx) in self.col_idxs.iter().enumerate() {
            if let Some(tf) = self.transforms.get(i) {
                let func = create_transform_function(tf)?;
                let arr_ref= arr_refs.get(col_idx.clone()).unwrap();
                let res_arr_ref = func.transform(arr_ref.to_owned())?;
                res_arr_refs.push(res_arr_ref);
            } else {
                return Err(anyhow!("not found transform for col idx {:?}", col_idx));
            }
        }
        
        return Ok(res_arr_refs);
    }
}


pub fn get_parition_key_from_first_val(partition_val_arr_refs: &Vec<ArrayRef>) -> anyhow::Result<PartitionKey> {
    let mut pk = Vec::<u8>::new();

    for arr_ref in partition_val_arr_refs {
        if arr_ref.is_empty() {
            return Err(anyhow!("get partition key with empty partition_val_arr_refs"));
        }

        match arr_ref.data_type() {
            DataType::Int32 => {
                let v = arr_ref.as_any().downcast_ref::<arrow::array::Int32Array>().unwrap().value(0);
                pk.extend_from_slice(&v.to_be_bytes());
            },
            DataType::Int64 => {
                let v = arr_ref.as_any().downcast_ref::<arrow::array::Int64Array>().unwrap().value(0);
                pk.extend_from_slice(&v.to_be_bytes());
            },
            DataType::Decimal128(_, _) => {
                let v = arr_ref.as_any().downcast_ref::<arrow::array::Decimal128Array>().unwrap().value(0);
                pk.extend_from_slice(&v.to_be_bytes());
            },
            DataType::Date32 => {
                let v = arr_ref.as_any().downcast_ref::<arrow::array::Date32Array>().unwrap().value(0);
                pk.extend_from_slice(&v.to_be_bytes());
            },
            DataType::Time64(TimeUnit::Microsecond) => {
                let v = arr_ref.as_any().downcast_ref::<arrow::array::Time64MicrosecondArray>().unwrap().value(0);
                pk.extend_from_slice(&v.to_be_bytes());
            },
            DataType::Timestamp(TimeUnit::Microsecond, _) => {
                let v = arr_ref.as_any().downcast_ref::<arrow::array::TimestampMicrosecondArray>().unwrap().value(0);
                pk.extend_from_slice(&v.to_be_bytes());
            },
            _ => {
                return Err(anyhow!("not support partition value data type for creating partition key"));
            }
        }
    }
    
    return Ok(pk);
}