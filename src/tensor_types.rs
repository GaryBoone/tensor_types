#[macro_export]
macro_rules! tensor_type {

    ($name:ident, [$($field:ident),*],  $params:ty, $kind:expr) => {

        pub struct $name {
            pub tensor: tch::Tensor,
        }

        impl $name {

            /// The new() function creates a new wrapper for a tensor. Its input is a tensor
            /// that will be checked for the required shape and a a parameters instance that
            /// contains values for the expected shape. It returns an instance of the new type
            /// defined with the tensor_type!() macro. It returns an error if the dimensions
            /// have not been initialized with the shape requirements or if the tensor does not
            /// have the required shape.
            ///
            /// # Example
            /// ```
            /// use tensor_types::tensor_type;
            /// use tch::Kind;
            /// pub struct Params {
            ///     size0: i64,
            ///     size1: i64,
            /// }
            /// tensor_type!(MyTensor, [size0, size1], Params, Kind::Float);
            ///
            /// let params = Params { size0: 2, size1: 3 };
            ///
            /// let tensor = tch::Tensor::from_slice(&[1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
            /// let wrapper = MyTensor::new(tensor, &params)?;
            ///
            /// assert_eq!(wrapper.tensor().size(), &[2, 3]);
            /// assert_eq!((*wrapper).size(), &[2, 3]);
            /// ```
            pub fn new(tensor: tch::Tensor, params: &$params) -> Result<Self, $crate::TensorTypeError> {
                let tensor_size = tensor.size();
                let expected_size: Vec<i64> = vec![$(params.$field.into()),*];

                if tensor_size != expected_size {
                    return Err($crate::TensorTypeError::ShapeMismatch {
                        type_name: stringify!($name).to_string(),
                        expected: expected_size,
                        found: tensor_size
                    });
                }

                if tensor.kind() != $kind {
                    return Err($crate::TensorTypeError::KindMismatch {
                        type_name: stringify!($name).to_string(),
                        expected: $kind,
                        found: tensor.kind()
                    });
                }
                Ok(Self { tensor })
            }

            /// The tensor() function returns a reference to the wrapped tensor.
            pub fn tensor(&self) -> &tch::Tensor { &self.tensor }

            /// The tensor_mut() function returns a mutable reference to the wrapped tensor.
            pub fn tensor_mut(&mut self) -> &mut tch::Tensor { &mut self.tensor }


            /// The apply_fn() function will apply a given function to the current value held by the
            /// newtype, returning another instance of the same newtype. The passed-in function is a
            /// closure that operates on a tch::Tensor and returns a tch::Tensor.
            /// Example:
            ///   let newAB_x2 = newAB.apply_fn(|t: &Tensor| t * 2, &params)?;
            pub fn apply_fn<F>(&self, tfn: F, params: &$params) -> Result<Self, $crate::TensorTypeError>
            where
                F: FnOnce(&tch::Tensor) -> tch::Tensor,
            {
                let transformed_tensor = tfn(&self.tensor);
                Self::new(transformed_tensor, params)
            }

            /// Note: cloning the tensor type creates a shallow clone of the underlying tensor.
            /// This is potentially confusing because tensor.clone() returns a deep clone.
            /// However, the newtype is a wrapper around a tensor, so cloning the newtype should
            /// clone the wrapper, not the data.
            pub fn clone(&self, params: &$params) -> Result<Self, $crate::TensorTypeError> {
                Self::new(self.tensor.shallow_clone(), params)
            }

            /// Unwrap the underlying tch::Tensor.
            pub fn into_inner(self) -> tch::Tensor {
                self.tensor
            }
        }


        impl std::fmt::Debug for $name {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                write!(f, "{:?}({:?})", stringify!($name), self.tensor)
            }
        }

        /// Implementing Deref allows the wrapped tch::Tensor to be dereferenced.
        impl std::ops::Deref for $name {
            type Target = tch::Tensor;

            fn deref(&self) -> &Self::Target {
                &self.tensor
            }
        }

    };
}

#[derive(thiserror::Error, Debug)]
pub enum TensorTypeError {
    #[error("shape mismatch on TensorType {type_name:?}: expected dimensions {expected:?}, found {found:?}")]
    ShapeMismatch {
        type_name: String,
        expected: Vec<i64>,
        found: Vec<i64>,
    },
    #[error(
        "kind mismatch on TensorType {type_name:?}: expected kind {expected:?}, found {found:?}"
    )]
    KindMismatch {
        type_name: String,
        expected: tch::Kind,
        found: tch::Kind,
    },
}
