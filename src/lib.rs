//! # Tensor Types
//!
//! The tensor_types macro allows the creation of typed, size-checked, kind-checked tensors,
//! providing:
//! 1. Tensor sizes are maintained and modified in a known way.
//! 1. The compiler provides type- and kind-checking for tensors.
//!
//! The types are created using the `tensor_type` macro.
//!
//! Usage:
//! 1. Create a new size-checked, kind-checked type for your tensors using the `tensor_type` macro.
//! ```rust
//! # pub struct Params {
//! #    size1: i64,
//! #    size2: i64,
//! #    size3: i64,
//! # }
//!     use tensor_types::tensor_type;
//!
//!     tensor_type!(MyTensor, [size1, size2, size3], Params, tch::Kind::Float);
//! ```
//! Here, `MyTensor` is the name of the new type that will wrap a tch::Tensor, `[size1, size2,
//! size3]` is a list of the fields in the `Params` type that gives the sizes of your tensor, and
//! `tch::Kind::Float` is the kind of the tensor.
//!
//! 2. The `Params` type can be defined however you like, so long as it provides the required
//! fields. So for example
//! ```rust
//!    pub struct Params {
//!       size1: i64,
//!       size2: i64,
//!       size3: i64,
//!    };
//! ```
//! The fields can be anything that can be cast to into an `i64`, which is the type used for
//! tch:Tensor dimensions.
//!  
//! The tensor_types crates also provides a `parameter_type` macro to make this easy and create
//! typed parameters, allowing the compiler to catch mixing up i64s. So, preferred, define types for
//! your dimension parameters using the `parameter_type` macro like this:
//! ```rust
//!     use tensor_types::{parameter_type, tensor_type};
//!
//!     parameter_type!(BatchSize, i64);
//!     parameter_type!(SequenceLength, i64);
//!     parameter_type!(ModelDimension, i64);
//!
//!     tensor_type!(MyTensor, [batch_size, sequence_length, model_dimension], Params, tch::Kind::Float);
//!
//!     pub struct Params {
//!        batch_size: BatchSize,
//!        sequence_length: SequenceLength,
//!        model_dimension: ModelDimension,
//!     };
//! ```
//!
//! 3. With these definitions in place, we can start using them at runtime. Somewhere near the start
//! of your program, instantiate the parameters structure so the tensor dimensions are available.
//! This happens at runtime for for loading from a configuration file and for flexibility in
//! testing.
//! ```rust
//! # use tensor_types::parameter_type;
//! # parameter_type!(BatchSize, i64);
//! # parameter_type!(SequenceLength, i64);
//! # parameter_type!(ModelDimension, i64);
//! # pub struct Params {
//! #    batch_size: BatchSize,
//! #    sequence_length: SequenceLength,
//! #    model_dimension: ModelDimension,
//! # };
//!     let params = Params {
//!        batch_size: BatchSize(1),
//!        sequence_length: SequenceLength(2),
//!        model_dimension: ModelDimension(3),
//!     };
//! ```
//!
//! 4. Now create wrapped instances of tch::Tensors using the `new` method. The size and kind will
//!    be checked by the new() method.
//!
//! ```rust
//! # use tensor_types::{parameter_type, tensor_type};
//! # parameter_type!(BatchSize, i64);
//! # parameter_type!(SequenceLength, i64);
//! # parameter_type!(ModelDimension, i64);
//! # #[derive(Default)]
//! # pub struct Params {
//! #    batch_size: BatchSize,
//! #    sequence_length: SequenceLength,
//! #    model_dimension: ModelDimension,
//! # }
//! # tensor_type!(MyTensor, [batch_size, sequence_length, model_dimension], Params, tch::Kind::Float);
//! # fn load_params_from_config_file() -> Result<Params, anyhow::Error> {
//! # Ok(
//! #     Params{batch_size: BatchSize(40), sequence_length: SequenceLength(100), model_dimension: ModelDimension(128)}
//! # )}
//! # fn main() -> Result<(), anyhow::Error> {
//!     let params = load_params_from_config_file()?;
//!
//!     //... tch::Tensor from somewhere.
//!     let tensor = tch::Tensor::randn([40, 100, 128], (tch::Kind::Float, tch::Device::Cpu));
//!     // Create an instance of MyTensor. It will be checked for size and kind, returning an error
//!     // if it's incorrect.
//!     let decoder_input = MyTensor::new(tensor, &params)?;
//!
//!     // Apply a tch::Tensor function to the wrapped tensor.
//!     // The result is size-checked and kind-checked again. The type is again MyTensor.
//!     let res = decoder_input.apply_fn(|t| t.triu(0), &params)?;
//!
//!     // Or access the wrapped tch::Tensor directly.
//!     assert_eq!(decoder_input.tensor().size(), &[40, 100, 128]);
//!     assert_eq!((*decoder_input).size(), &[40, 100, 128]);
//! # Ok(())
//! # }

pub use tensor_types::TensorTypeError;

mod parameter_types;
mod tensor_types;
