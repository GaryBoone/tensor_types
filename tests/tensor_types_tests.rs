#[cfg(test)]
mod tests {
    use anyhow::Result;
    use tch::Tensor;
    use tensor_types::{parameter_type, tensor_type};

    #[test]
    fn test_uninitialized_error() {
        parameter_type!(MyParam1, i64);
        parameter_type!(MyParam2, i64);
        tensor_type!(MyTensor, [MyParam1, MyParam2, MyParam1]);

        // Error to call new() before set().
        let t = Tensor::randn([1, 2, 3], (tch::Kind::Float, tch::Device::Cpu));
        match MyTensor::new(t) {
            Err(MyTensorError::Uninitialized { type_name: _ }) => (),
            _ => panic!("expected Uninitialized"),
        };
    }

    #[test]
    fn test_multiple_set() {
        parameter_type!(MyParam1, i64);
        parameter_type!(MyParam2, i64);
        tensor_type!(MyTensor, [MyParam1, MyParam2, MyParam1]);
        MyTensor::set(MyParam1(1), MyParam2(2), MyParam1(3)).unwrap();

        // Error to call set() more than once.
        match MyTensor::set(MyParam1(1), MyParam2(2), MyParam1(3)) {
            Err(MyTensorError::AlreadyInitialized { type_name: _ }) => (),
            _ => panic!("expected AlreadyInitialized"),
        };
    }

    #[test]
    fn test_wrong_size() {
        parameter_type!(MyParam1, i64);
        parameter_type!(MyParam2, i64);
        tensor_type!(MyTensor, [MyParam1, MyParam2, MyParam1]);
        MyTensor::set(MyParam1(1), MyParam2(2), MyParam1(3)).unwrap();

        // Error to call new() with wrong-sized tensor.
        let t = Tensor::randn([1, 2, 1], (tch::Kind::Float, tch::Device::Cpu));
        match MyTensor::new(t) {
            Err(MyTensorError::ShapeMismatch {
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
    fn test_basic() {
        parameter_type!(MyParam1, i64);
        parameter_type!(MyParam2, i64);
        tensor_type!(MyTensor, [MyParam1, MyParam2, MyParam1]);
        MyTensor::set(MyParam1(1), MyParam2(2), MyParam1(3)).unwrap();

        let t = Tensor::randn([1, 2, 3], (tch::Kind::Float, tch::Device::Cpu));
        let my_tensor = MyTensor::new(t).unwrap();

        // Ok to call new() again with right-sized tensor to create another instance.
        let t2 = Tensor::randn([1, 2, 3], (tch::Kind::Float, tch::Device::Cpu));
        assert!(MyTensor::new(t2).is_ok());

        assert_eq!(my_tensor.tensor().size(), &[1, 2, 3]);

        // Create a second TensorType, confirming independence of dimensions.
        tensor_type!(MyTensor2, [MyParam1, MyParam2, MyParam1, MyParam2]);
        MyTensor2::set(MyParam1(1), MyParam2(2), MyParam1(3), MyParam2(3)).unwrap();
        let t2 = Tensor::randn([1, 2, 3, 3], (tch::Kind::Float, tch::Device::Cpu));
        let my_tensor2 = MyTensor2::new(t2).unwrap();
        assert_eq!(my_tensor2.tensor().size(), &[1, 2, 3, 3]);
    }

    #[test]
    fn test_types() {
        parameter_type!(MyParam1, i64);
        parameter_type!(MyParam2, i64);
        tensor_type!(MyTensor, [MyParam1, MyParam2, MyParam1]);
        MyTensor::set(MyParam1(1), MyParam2(2), MyParam1(3)).unwrap();
        let t = Tensor::randn([1, 2, 3], (tch::Kind::Float, tch::Device::Cpu));
        let my_tensor = MyTensor::new(t).unwrap();

        fn type_of<T>(_: T) -> &'static str {
            std::any::type_name::<T>()
        }

        assert_eq!(
            type_of(&my_tensor),
            "&tensor_types_tests::tests::test_types::MyTensor"
        );
        assert_eq!(type_of(&*my_tensor), "&tch::wrappers::tensor::Tensor");
    }

    #[test]
    fn test_accessors() {
        parameter_type!(MyParam1, i64);
        parameter_type!(MyParam2, i64);
        tensor_type!(MyTensor, [MyParam1, MyParam2, MyParam1]);
        MyTensor::set(MyParam1(1), MyParam2(2), MyParam1(3)).unwrap();
        let t = Tensor::randn([1, 2, 3], (tch::Kind::Float, tch::Device::Cpu));
        let my_tensor = MyTensor::new(t).unwrap();

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
        tensor_type!(MyTensor, [i64, i64]);
        MyTensor::set(2, 3).unwrap();
        let tensor = tch::Tensor::from_slice(&[0, 1, 2, 3, 4, 5]).reshape([2, 3]);
        let my_tensor = MyTensor::new(tensor).unwrap();
        // clone() makes a clone of MyTensor with a shallow_clone of the wrapped tensor.
        let cloned_wrapper = my_tensor.clone();

        assert_eq!(cloned_wrapper.size(), vec![2, 3]);
    }

    #[test]
    fn test_apply() {
        parameter_type!(MyParam1, i64);
        parameter_type!(MyParam2, i64);
        tensor_type!(MyTensor, [MyParam1, MyParam2, MyParam1]);
        MyTensor::set(MyParam1(1), MyParam2(2), MyParam1(3)).unwrap();
        let t = Tensor::randn([1, 2, 3], (tch::Kind::Float, tch::Device::Cpu));
        let my_tensor = MyTensor::new(t).unwrap();

        // Test the apply() method.
        let new_my_tensor = my_tensor.apply(|t| t.fill(1.0)).unwrap();
        assert_eq!(new_my_tensor.tensor().size(), &[1, 2, 3]);
        let tensor_of_ones = Tensor::ones([1, 2, 3], (tch::Kind::Float, tch::Device::Cpu));
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

        // But it's an error if the function given to apply changes the size of the wrapped tensor.
        // The tensor shape is fixed and is checked.
        match my_tensor.apply(|t| t.transpose(1, 2)) {
            Err(MyTensorError::ShapeMismatch {
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
}
