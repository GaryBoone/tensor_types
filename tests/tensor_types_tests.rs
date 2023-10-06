#[cfg(test)]
mod tests {
    use anyhow::Result;
    use tch::{Device, Kind, Tensor};
    use tensor_types::{parameter_type, tensor_type, TensorType, TensorTypeError};

    // This test shows the basic, correct usage of the parameter_type and tensor_type macros.
    #[test]
    fn test_basic() {
        // 1. Somewhere in your code, define the types your program will use.
        // Define a parameter type we'll use to specify the dimensionality of our tensors.
        parameter_type!(MyParam1, i64);
        parameter_type!(MyParam2, i64);
        parameter_type!(MyParam3, i64);
        // Define a parameter structure for the runtime values your tensor types will use.
        pub struct Params {
            my_param1: MyParam1,
            my_param2: MyParam2,
            my_param3: MyParam3,
        }
        // Define a TensorType with a unique name, telling it which parameters to use for size
        // checking.
        tensor_type!(
            MyTensor,
            [my_param1, my_param2, my_param3],
            Params,
            Kind::Float
        );

        // 2. At some point near the start of your program, instantiate the parameters structure to
        // given them runtime values. For example, these may be read from a config file.
        let params = Params {
            my_param1: MyParam1(1),
            my_param2: MyParam2(2),
            my_param3: MyParam3(3),
        };

        // 3. Now throughout your program, you can create instance of your tensor type that will
        // be checked for size and kind correctness.
        let t = Tensor::randn([1, 2, 3], (Kind::Float, Device::Cpu));
        let my_tensor = MyTensor::new(t, &params).unwrap();

        // Call new() to create another instance, again checking the size and kind.
        let t2 = Tensor::randn([1, 2, 3], (Kind::Float, Device::Cpu));
        assert!(MyTensor::new(t2, &params).is_ok());

        // You can access the tensor with the tensor() method.
        assert_eq!(my_tensor.tensor().size(), &[1, 2, 3]);
        // Or by dereferencing it.
        assert_eq!((*my_tensor).size(), &[1, 2, 3]);

        // Create a second TensorType with different size and kind.
        tensor_type!(
            MyTensor2,
            [my_param1, my_param2, my_param1, my_param2], // Note repeated dimensions. That's ok.
            Params,
            Kind::Int64
        );
        let t2 = Tensor::from_slice(&[1, 2, 3, 4])
            .reshape([1, 2, 1, 2])
            .to_kind(Kind::Int64);
        let my_tensor2 = MyTensor2::new(t2, &params).unwrap();
        assert_eq!(my_tensor2.tensor().size(), &[1, 2, 1, 2]);
    }

    // For the other tests, define these globally.
    parameter_type!(MyParam1, i64);
    parameter_type!(MyParam2, i64);
    parameter_type!(MyParam3, i64);
    pub struct Params {
        my_param1: MyParam1,
        my_param2: MyParam2,
        my_param3: MyParam3,
    }
    tensor_type!(
        MyTensor,
        [my_param1, my_param2, my_param3],
        Params,
        Kind::Float
    );
    fn setup() -> Params {
        Params {
            my_param1: MyParam1(1),
            my_param2: MyParam2(2),
            my_param3: MyParam3(3),
        }
    }

    #[test]
    fn test_wrong_size() {
        let params = setup();

        // It's an error to call new() with a wrongly-dimensioned tensor.
        let t = Tensor::randn([1, 2], (Kind::Float, Device::Cpu));
        match MyTensor::new(t, &params) {
            Err(TensorTypeError::ShapeMismatch {
                type_name,
                expected,
                found,
            }) => {
                if type_name != "MyTensor" || expected != vec![1, 2, 3] || found != vec![1, 2] {
                    panic!("expected ShapeMismatch, but unexpected type_name ({}), found ({:?}) or expected ({:?})", type_name, found, expected)
                }
            }
            _ => panic!("expected ShapeMismatch"),
        };

        // It's an error to call new() with a wrongly-sized tensor.
        let t = Tensor::randn([1, 2, 1], (Kind::Float, Device::Cpu));
        match MyTensor::new(t, &params) {
            Err(TensorTypeError::ShapeMismatch {
                type_name,
                expected,
                found,
            }) => {
                if type_name != "MyTensor" || expected != vec![1, 2, 3] || found != vec![1, 2, 1] {
                    panic!("expected ShapeMismatch, but unexpected type_name ({}), found ({:?}) or expected ({:?})", type_name, found, expected)
                }
            }
            _ => panic!("expected ShapeMismatch"),
        };
    }

    #[test]
    fn test_wrong_kind() {
        let params = setup();

        // It's an error to call new() with the wrong kind of tensor.
        let t = Tensor::from_slice(&[1, 2, 3, 4, 5, 6])
            .reshape([1, 2, 3])
            .to_kind(Kind::Int64);
        match MyTensor::new(t, &params) {
            Err(TensorTypeError::KindMismatch {
                type_name,
                expected,
                found,
            }) => {
                if type_name != "MyTensor" || expected != Kind::Float || found != Kind::Int64 {
                    panic!("expected ShapeMismatch, but unexpected type_name ({}), found ({:?}) or expected ({:?})", type_name, found, expected)
                }
            }
            _ => panic!("expected KindMismatch"),
        };
    }

    #[test]
    fn test_types() {
        let params = setup();
        let t = Tensor::randn([1, 2, 3], (Kind::Float, Device::Cpu));
        let my_tensor = MyTensor::new(t, &params).unwrap();

        fn type_of<T>(_: T) -> &'static str {
            std::any::type_name::<T>()
        }

        // The type of my_tensor is MyTensor.
        assert_eq!(type_of(&my_tensor), "&tensor_types_tests::tests::MyTensor");
        // The type of what it wraps is Tensor.
        assert_eq!(type_of(&*my_tensor), "&tch::wrappers::tensor::Tensor");
    }

    #[test]
    fn test_accessors() {
        let params = setup();
        let t = Tensor::randn([1, 2, 3], (Kind::Float, Device::Cpu));
        let my_tensor = MyTensor::new(t, &params).unwrap();

        // Access the tensor using the tensor() function.
        assert_eq!(my_tensor.tensor().size(), &[1, 2, 3]);

        // Access the tensor using explicit deref.
        assert_eq!((*my_tensor).size(), &[1, 2, 3]);

        // Access the tensor using automatic deref.
        assert_eq!(my_tensor.size(), &[1, 2, 3]);

        // Unwrap the inner tensor.
        let unwrapped_tensor = my_tensor.into_inner();
        assert_eq!(unwrapped_tensor.size(), &[1, 2, 3]);
        fn type_of<T>(_: T) -> &'static str {
            std::any::type_name::<T>()
        }
        assert_eq!(type_of(&unwrapped_tensor), "&tch::wrappers::tensor::Tensor");
    }

    #[test]
    fn test_clone() {
        let params = setup();
        tensor_type!(MyTensor2, [my_param2, my_param3], Params, Kind::Int);
        let tensor = Tensor::from_slice(&[0, 1, 2, 3, 4, 5]).reshape([2, 3]);
        let my_tensor2 = MyTensor2::new(tensor, &params).unwrap();
        // clone() makes a clone of MyTensor with a shallow_clone of the wrapped tensor.
        let cloned_wrapper = my_tensor2.clone(&params).unwrap();

        assert_eq!((*cloned_wrapper).size(), vec![2, 3]);
    }

    #[test]
    fn test_apply_fn() {
        let params = setup();
        let t = Tensor::randn([1, 2, 3], (Kind::Float, Device::Cpu));
        let my_tensor = MyTensor::new(t, &params).unwrap();

        // Test the apply_fn() method.
        let new_my_tensor = my_tensor.apply_fn(|t| t.fill(1.0), &params).unwrap();
        assert_eq!(new_my_tensor.tensor().size(), &[1, 2, 3]);
        let tensor_of_ones = Tensor::ones([1, 2, 3], (Kind::Float, Device::Cpu));
        assert_eq!(
            new_my_tensor
                .eq_tensor(&tensor_of_ones)
                .all()
                .int64_value(&[]),
            1
        );
        assert_eq!(
            tensor_of_ones
                .eq_tensor(&new_my_tensor)
                .all()
                .int64_value(&[]),
            1
        );

        // But it's an error if the function given to apply_fn changes the size of the wrapped tensor.
        // The tensor shape is fixed and is checked.
        match my_tensor.apply_fn(|t| t.transpose(1, 2), &params) {
            Err(TensorTypeError::ShapeMismatch {
                type_name,
                expected,
                found,
            }) => {
                if type_name != "MyTensor" || expected != vec![1, 2, 3] || found != vec![1, 3, 2] {
                    panic!("expected ShapeMismatch, but unexpected type_name ({}), found ({:?}) or expected ({:?})", type_name, found, expected)
                }
            }
            _ => panic!("expected ShapeMismatch"),
        };
    }

    #[test]
    fn test_trait_bounds() {
        pub trait AttentionTensorTrait {}
        tensor_type!(
            BatchSeqDModelTensor,
            [batch_size, sequence_length, d_model],
            Params,
            Kind::Float
        );
        tensor_type!(
            BatchSeqDReducedTensor,
            [batch_size, sequence_length, d_reduced],
            Params,
            Kind::Float
        );

        // Attach the AttentionTensorTrait to our types.
        impl AttentionTensorTrait for BatchSeqDModelTensor {}
        impl AttentionTensorTrait for BatchSeqDReducedTensor {}

        parameter_type!(BatchSize, i64);
        parameter_type!(SequenceLength, i64);
        parameter_type!(DModel, i64);
        parameter_type!(DReduced, i64);
        pub struct Params {
            batch_size: BatchSize,
            sequence_length: SequenceLength,
            d_model: DModel,
            d_reduced: DReduced,
        }
        let params = Params {
            batch_size: BatchSize(1),
            sequence_length: SequenceLength(2),
            d_model: DModel(3),
            d_reduced: DReduced(4),
        };

        fn attention<T: TensorType<InnerType = Params> + AttentionTensorTrait>(
            query: &T,
            params: &Params,
        ) -> Result<T, TensorTypeError> {
            // ... do something with the tensors ...
            query.apply_fn(|t| t.triu(1), params)
        }

        let t = Tensor::randn([1, 2, 3], (Kind::Float, Device::Cpu));
        let query = BatchSeqDModelTensor::new(t, &params).unwrap();
        let _ = attention(&query, &params).unwrap();

        let t = Tensor::randn([1, 2, 4], (Kind::Float, Device::Cpu));
        let query = BatchSeqDReducedTensor::new(t, &params).unwrap();
        let _ = attention(&query, &params).unwrap();
    }
}
