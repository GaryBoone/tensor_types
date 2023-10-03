//! # Tensor Types
//!
//! The tensor_types allows the creation of typed, size-checked, kind-checked tensors, providing:
//! 1. Tensor sizes are maintained and modified in a known way.
//! 1. The compiler provides type- and kind-checking for tensors.
//!
//! The types are created using the `tensor_type` macro.
//!
//! Usage:
//! 1. Create a new size-checked, kind-checked type for your tensors using the `tensor_type` macro.
//! ```rust
//!     use tensor_types::tensor_type;
//!
//!     tensor_type!(MyTensor, [i64, i64, i64], tch::Kind::Float);
//! ```
//! Or, preferred, define types for your dimension parameters using the `parameter_type` macro.
//! ```rust
//!     use tensor_types::{parameter_type, tensor_type};
//!
//!     parameter_type!(BatchSize, i64);
//!     parameter_type!(SequenceLength, i64);
//!     parameter_type!(ModelDimension, i64);
//!
//!     tensor_type!(MyTensor, [BatchSize, SequenceLength, ModelDimension], tch::Kind::Float);
//! ```
//!
//! 2. Set the dimensions of your tensor using the `set` method, perhaps after reading them from a
//!    configuration file or from a model.
//!
//! ```rust
//! # use tensor_types::{parameter_type, tensor_type};
//! # parameter_type!(BatchSize, i64);
//! # parameter_type!(SequenceLength, i64);
//! # parameter_type!(ModelDimension, i64);
//!
//! # fn main() -> Result<(), anyhow::Error> {
//!     tensor_type!(MyTensor, [BatchSize, SequenceLength, ModelDimension], tch::Kind::Float);
//!     MyTensor::set(BatchSize(40), SequenceLength(100), ModelDimension(128))?;
//! # Ok(())
//! # }
//! ```
//!
//! 3. Now create wrapped instances of tch::Tensors using the `new` method. The size will be checked
//!    by the new() method.
//!
//! ```rust
//! # use tensor_types::{parameter_type, tensor_type};
//! # parameter_type!(BatchSize, i64);
//! # parameter_type!(SequenceLength, i64);
//! # parameter_type!(ModelDimension, i64);
//! # tensor_type!(MyTensor, [BatchSize, SequenceLength, ModelDimension], tch::Kind::Float);
//!
//! # fn main() -> Result<(), anyhow::Error> {
//!     # MyTensor::set(BatchSize(40), SequenceLength(100), ModelDimension(128))?;
//!     //... tch::Tensor from somewhere.
//!     let tensor = tch::Tensor::randn([40, 100, 128], (tch::Kind::Float, tch::Device::Cpu));
//!     let decoder_input = MyTensor::new(tensor)?;
//!
//!     // Apply a tch::Tensor function to the wrapped tensor.
//!     // The result is size-checked and kind-checked again. The type is MyTensor.
//!     let res = decoder_input.apply(|t| t.triu(0))?;
//!
//!     // Or access the wrapped tch::Tensor directly.
//!     assert_eq!(decoder_input.tensor().size(), &[40, 100, 128]);
//!     assert_eq!((*decoder_input).size(), &[40, 100, 128]);
//! # Ok(())
//! # }
pub use tensor_types_proc::tensor_type;

mod parameter_types;
