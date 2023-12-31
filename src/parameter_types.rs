/// The parameter_type! macro defines a new named type with a specific inner type.
///
/// This macro generates a new struct type with the specified name and inner type, along with
/// implementations for various traits such as accessors and display.
///
/// # Arguments
///
/// * `$type_name` - The name of the new parameter type.
/// * `$inner_type` - The inner type of the new parameter type.
///
/// # Example
///
/// ```
/// use tensor_types::parameter_type;
///
/// parameter_type!(MyParam, i64);
///
/// let param = MyParam(42);
/// assert_eq!(*param, 42);
/// assert_eq!(i64::from(param), 42i64);
/// ```
/// Copy is also implemented for each newtype (which requires Clone), so that they can be passed by
/// value. Debug is commonly required, such as by the tch::nn::ModuleT trait. Hash is provided so
/// that these types can be used as keys in a HashMap. Serialize and Deserialize are provided so
/// that the model can be saved and loaded.
#[macro_export]
macro_rules! parameter_type {
    ($type_name:ident, $inner_type:ty) => {
        #[derive(
            Debug, Clone, Copy, PartialEq, PartialOrd, serde::Serialize, serde::Deserialize, Hash,
        )]
        pub struct $type_name(pub $inner_type); // TODO: remove pub?

        /// Implements the conversion from an i64 value to the specified parameter type.
        impl From<i64> for $type_name {
            fn from(val: i64) -> Self {
                $type_name(val)
            }
        }

        /// Converts a value of type `$type_name` to an `i64`.
        impl From<$type_name> for i64 {
            fn from(val: $type_name) -> Self {
                val.0
            }
        }

        /// This implementation provides a dereferencing mechanism for the `$type_name` type.
        /// It allows the inner `$inner_type` to be accessed through a reference.
        impl std::ops::Deref for $type_name {
            type Target = $inner_type;

            /// Returns a reference to the inner `$inner_type`.
            fn deref(&self) -> &$inner_type {
                &self.0
            }
        }

        /// Implements the DerefMut trait for the given type.
        impl std::ops::DerefMut for $type_name {
            fn deref_mut(&mut self) -> &mut $inner_type {
                &mut self.0
            }
        }

        /// Implements the `AsRef` trait for the specified `$type_name` type, allowing it to be
        /// referenced as an `i64`.
        impl AsRef<i64> for $type_name {
            /// Returns a reference to the `i64` value contained within the `$type_name` instance.
            fn as_ref(&self) -> &i64 {
                &self.0
            }
        }

        /// Implements the Display trait for the given type.
        impl std::fmt::Display for $type_name {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                use num_format::ToFormattedString;
                write!(f, "{}", self.0.to_formatted_string(&num_format::Locale::en))
            }
        }

        // Implements the Default trait for the given type.
        impl Default for $type_name {
            fn default() -> Self {
                $type_name(Default::default())
            }
        }

        // TODO: Remove.
        impl $type_name {
            pub fn v(&self) -> $inner_type {
                self.0
            }
        }
    };
}
