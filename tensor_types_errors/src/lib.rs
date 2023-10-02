use thiserror::Error;

/// TensorTypeError provides the errors for TensorTypes created by the `tensor_type!` macro.
#[derive(Error, Debug)]
pub enum TensorTypeError {
    #[error("shape mismatch on TensorType {type_name:?}: expected dimensions {expected:?}, found {found:?}")]
    ShapeMismatchError {
        type_name: String,
        expected: Vec<i64>,
        found: Vec<i64>,
    },
    #[error("new() called on uninitialized TensorType {type_name:?}")]
    UninitializedError { type_name: String },
    #[error("set() called on already initialized TensorType {type_name:?}")]
    AlreadyInitializedError { type_name: String },
}
