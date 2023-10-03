//! This crate provides the tensor_type macro to the tensor_types crate.
//!
//! Note: Do not use this crate directly. Instead, use the
//! [tensor_types](https://crates.io/crates/tensor_types) crate. See the repository at
//! [github.com/GaryBoone/tensor_types](https://github.com/GaryBoone/tensor_types) for more information.
//!
//! This crate exists as a standalone crate because of the way proc_macros work in Rust. In
//! addition, Crates.io does not recognize sub-crates, so this crate appears as a standalone crate.
use proc_macro::TokenStream;
use quote::quote;
use syn::parse::{Parse, ParseStream};
use syn::punctuated::Punctuated;
use syn::token::Comma;
use syn::{parse_macro_input, Ident, Type};

// From the macro input, extract the name of the struct and vector of types.
struct MacroInput {
    name: Ident,
    types: Vec<Type>,
    kind: syn::Path,
}

impl Parse for MacroInput {
    fn parse(input: ParseStream) -> syn::Result<Self> {
        // Parse the name of the struct.
        let name: Ident = input.parse()?;

        // Parse the list of types as a bracketed, comma-separated list.
        input.parse::<syn::Token![,]>()?;
        let content;
        syn::bracketed!(content in input);
        let types = Punctuated::<Type, Comma>::parse_terminated(&content)?;
        let types = types.into_iter().collect();

        // Parse the tch::Kind
        input.parse::<syn::Token![,]>()?;
        let kind: syn::Path = input.parse()?;

        Ok(MacroInput { name, types, kind })
    }
}

/// The `tensor_type!` macro provides a macro for generating strongly-typed, shape-checked tensor
/// wrappers for `tch::Tensor`s.
///
/// The macro generates a struct that wraps a `tch::Tensor`, allows the required shape to be set,
/// and checks that wrapped tensor meet the shape requirements. It also provides methods accessing
/// the tensor and applying tensor operations to the tensor.
///
/// To use the macro, 1) define the new type using the tensor_type! macro. It takes the new type's
/// name followed by a list of types that provide the required dimensions of the tensor. 2) Call
/// the new type's `set()` function with the actual values of the tensor's dimensions. 3) Call the
/// constructor with a `tch::Tensor` that meets the required shape. 3) Use the new type's
/// methods to access the tensor and apply tensor operations to the tensor.
///
/// # Example
/// ```rust
/// # use tensor_types_proc::tensor_type;
/// # use tch::Kind;
/// tensor_type!(MyTensor, [i64, i64], Kind::Int);
///
/// MyTensor::set(2, 3).unwrap();
/// let tensor = tch::Tensor::from_slice(&[1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
/// let wrapper = MyTensor::new(tensor).unwrap();
/// assert_eq!(wrapper.tensor().size(), &[2, 3]);
/// assert_eq!((*wrapper).size(), &[2, 3]);
/// ```

#[proc_macro]
pub fn tensor_type(input: TokenStream) -> TokenStream {
    let MacroInput { name, types, kind } = parse_macro_input!(input as MacroInput);

    // Generate unique variable names like x0, x1, etc.
    let vars: Vec<Ident> = (0..types.len())
        .map(|i| Ident::new(&format!("x{}", i), proc_macro2::Span::call_site()))
        .collect();

    let types_len = types.len();

    // We'll create a module named `<name>_module` to hold the static variables, isolating them.
    let module_name = Ident::new(&format!("{}_module", &name), proc_macro2::Span::call_site());

    let error_type_name = Ident::new(&format!("{}Error", &name), proc_macro2::Span::call_site());

    // The user may input `tch::Kind::Float` or just `Kind::Float`, so we need to just store the
    // variant and construct the full path ourselves.
    let kind_variant = kind.segments.last().unwrap().ident.clone();

    let expanded = quote! {

        pub mod #module_name {
            use std::sync::{Once, Mutex};

            pub static INIT: Once = Once::new();
            pub static DIMS: Mutex<[i64; #types_len]> = Mutex::new([0; #types_len]);
            pub static KIND: Mutex<tch::Kind> = Mutex::new(tch::Kind::#kind_variant);
        }

        #[derive(Debug)]
        pub struct #name {
            tensor: tch::Tensor,
        }

        impl #name {
            /// Checks if the dimensions have been initialized.
            ///
            /// # Example
            /// ```
            /// # use tensor_types::tensor_type;
            /// # use tch::Kind;
            /// # tensor_type!(MyTensor, [i64, i64], Kind::Float);
            /// assert_eq!(MyTensor::is_initialized(), false);
            /// ```
            pub fn is_initialized() -> bool {
                #module_name::INIT.is_completed()
            }

            /// Returns the dimensions of the tensor if initialized.
            ///
            /// # Example
            /// ```
            /// # use tensor_types::tensor_type;
            /// # use tch::Kind;
            /// # tensor_type!(MyTensor, [i64, i64], Kind::Float);
            /// # MyTensor::set(2, 3).unwrap();
            /// assert_eq!(MyTensor::get_dims(), Some(vec![2, 3]));
            /// ```
            pub fn get_dims() -> Option<Vec<i64>> {
                if Self::is_initialized() {
                    Some(#module_name::DIMS.lock().unwrap().to_vec())
                } else {
                    None
                }
            }

            /// Returns the kind of the tensor.
            ///
            /// # Example
            /// ```
            /// # use tensor_types::tensor_type;
            /// # use tch::Kind;
            /// # tensor_type!(MyTensor, [i64, i64], Kind::Int64);
            /// assert_eq!(MyTensor::get_kind(), Some(Kind::Int64));
            /// ```
            pub fn get_kind() -> Option<tch::Kind> {
                if Self::is_initialized() {
                    Some(*#module_name::KIND.lock().unwrap())
                } else {
                    None
                }
            }

            /// Sets the required dimensions of the wrapped tensor. Its input is a vector of values
            /// of the required dimensions. The vector must be the same length as the number of
            /// dimensions defined when the new type was created with the tensor_type!() macro. It
            /// can only be called once. Returns an error if the dimensions have already been
            /// initialized.
            ///
            /// # Example
            /// ```
            /// # use tensor_types::tensor_type;
            /// # use tch::Kind;
            /// # tensor_type!(MyTensor, [i64, i64] Kind::Float);
            /// # MyTensor::set(2, 3).unwrap();
            /// assert_eq!(MyTensor::get_dims(), Some(vec![2, 3]));
            /// ```
            pub fn set(#( #vars: #types, )*) -> Result<(), #error_type_name>
            where
                #( #types: Into<i64>, )*
            {
                if #module_name::INIT.is_completed() {
                    return Err(#error_type_name::AlreadyInitialized {
                        type_name: stringify!(#name).to_string()
                    });
                }
                #module_name::INIT.call_once(|| {
                    let mut dims = #module_name::DIMS.lock().unwrap();
                    *dims = [ #( #vars.into(), )* ];
                });
                Ok(())
            }

            /// The new() function creates a new wrapper for a tensor. Its input is a tensor that
            /// will be checked for the required shape. Returns an instance of the new type defined
            /// with the tensor_type!() macro. Returns an error if the dimensions have not been
            /// initialized with the shape requirements or if the tensor does not have the required
            /// shape.
            ///
            /// # Example
            /// ```
            /// # use tensor_types::tensor_type;
            /// # use tch::Kind;
            /// # tensor_type!(MyTensor, [i64, i64] Kind::Float);
            /// # MyTensor::set(2, 3).unwrap();
            /// assert_eq!(MyTensor::get_dims(), Some(vec![2, 3]));
            /// let tensor = tch::Tensor::from_slice(&[1, 2, 3, 4, 5, 6]).reshape(&[2, 3]);
            /// let wrapper = MyTensor::new(tensor).unwrap();
            /// assert_eq!(wrapper.tensor().size(), &[2, 3]);
            /// assert_eq!((*wrapper).size(), &[2, 3]);
            /// ```
            pub fn new(tensor: tch::Tensor) -> Result<Self, #error_type_name> {
                if !#module_name::INIT.is_completed() {
                    return Err(#error_type_name::Uninitialized {
                        type_name: stringify!(#name).to_string()
                    });
                }

                // Check kind.
                let kind = #module_name::KIND.lock().unwrap();
                if *kind != tensor.kind() {
                    return Err(#error_type_name::KindMismatch {
                        type_name: stringify!(#name).to_string(),
                        expected: *kind,
                        found: tensor.kind()
                    });
                }

                // Check size.
                let size = tensor.size();
                let dims = #module_name::DIMS.lock().unwrap();
                if size != dims.to_vec() {
                    return Err(#error_type_name::ShapeMismatch {
                        type_name: stringify!(#name).to_string(),
                        expected: dims.to_vec(),
                        found: size.to_vec()
                    });
                }
                Ok(Self { tensor })
            }

            pub fn tensor(&self) -> &tch::Tensor {
                &self.tensor
            }

            pub fn into_inner(self) -> tch::Tensor {
                self.tensor
            }

            /// Returns a shallow clone of the tensor wrapper.
            ///
            /// # Examples
            ///
            /// ```rust
            /// # use tensor_types::tensor_type;
            /// # use tch::Kind;
            /// # tensor_type!(YourType, i64, i64], Kind::Float);
            /// let _ = YourType::set(2, 3);
            /// let tensor = tch::Tensor::of_size(&[2, 3]);
            /// let wrapper = YourType::new(tensor).unwrap();
            /// let cloned_wrapper = wrapper.clone();
            /// assert_eq!(cloned_wrapper.shape(), vec![2, 3]);
            pub fn clone(&self) -> Self {
                Self { tensor: self.tensor.shallow_clone() }
            }

            pub fn apply<F>(&self, f: F) -> Result<Self, #error_type_name>
            where
                F: Fn(&tch::Tensor) -> tch::Tensor,
            {
                Self::new(f(&self.tensor))
            }
        }


        /// The Errors returned by the TensorType are named like <type_name>Error and provide the
        /// errors for TensorTypes created by the `tensor_type!` macro.
        #[derive(thiserror::Error, Debug)]
        pub enum #error_type_name {
            #[error("shape mismatch on TensorType {type_name:?}: expected dimensions {expected:?}, found {found:?}")]
            ShapeMismatch {
                type_name: String,
                expected: Vec<i64>,
                found: Vec<i64>,
            },
            #[error("kind mismatch on TensorType {type_name:?}: expected kind {expected:?}, found {found:?}")]
            KindMismatch {
                type_name: String,
                expected: tch::Kind,
                found: tch::Kind,
            },
            #[error("new() called on uninitialized TensorType {type_name:?}")]
            Uninitialized { type_name: String },
            #[error("set() called on already initialized TensorType {type_name:?}")]
            AlreadyInitialized { type_name: String },
        }

        /// Implementing Deref allows the wrapped tch::Tensor to be dereferenced.
        impl std::ops::Deref for #name {
            type Target = tch::Tensor;

            fn deref(&self) -> &Self::Target {
                &self.tensor
            }
        }
    };

    TokenStream::from(expanded)
}
