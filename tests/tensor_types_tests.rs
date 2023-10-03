#[cfg(test)]
mod tests {
    use anyhow::Result;
    use tch::{Device, Kind, Tensor};
    use tensor_types::{parameter_type, tensor_type};

    #[test]
    fn test_uninitialized_error() {
        parameter_type!(MyParam1, i64);
        parameter_type!(MyParam2, i64);
        tensor_type!(MyTensor, [MyParam1, MyParam2, MyParam1], Kind::Float);

        // Error to call new() before set().
        let t = Tensor::randn([1, 2, 3], (Kind::Float, Device::Cpu));
        match MyTensor::new(t) {
            Err(MyTensorError::Uninitialized { type_name: _ }) => (),
            _ => panic!("expected Uninitialized"),
        };
    }

    #[test]
    fn test_multiple_set() {
        parameter_type!(MyParam1, i64);
        parameter_type!(MyParam2, i64);
        tensor_type!(MyTensor, [MyParam1, MyParam2, MyParam1], Kind::Float);
        MyTensor::set(MyParam1(1), MyParam2(2), MyParam1(3));
        assert_eq!(MyTensor::get_dims(), Some(vec![1, 2, 3]));
        MyTensor::set(MyParam1(3), MyParam2(1), MyParam1(2));
        assert_eq!(MyTensor::get_dims(), Some(vec![3, 1, 2]));
    }

    #[test]
    fn test_wrong_size() {
        parameter_type!(MyParam1, i64);
        parameter_type!(MyParam2, i64);
        tensor_type!(MyTensor, [MyParam1, MyParam2, MyParam1], Kind::Float);
        MyTensor::set(MyParam1(1), MyParam2(2), MyParam1(3));

        // Error to call new() with wrong-sized tensor.
        let t = Tensor::randn([1, 2, 1], (Kind::Float, Device::Cpu));
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
        tensor_type!(MyTensor, [MyParam1, MyParam2, MyParam1], Kind::Float);
        MyTensor::set(MyParam1(1), MyParam2(2), MyParam1(3));

        let t = Tensor::randn([1, 2, 3], (Kind::Float, Device::Cpu));
        let my_tensor = MyTensor::new(t).unwrap();

        // Ok to call new() again with right-sized tensor to create another instance.
        let t2 = Tensor::randn([1, 2, 3], (Kind::Float, Device::Cpu));
        assert!(MyTensor::new(t2).is_ok());

        assert_eq!(my_tensor.tensor().size(), &[1, 2, 3]);

        // Create a second TensorType, confirming independence of dimensions.
        tensor_type!(
            MyTensor2,
            [MyParam1, MyParam2, MyParam1, MyParam2],
            Kind::Float
        );
        MyTensor2::set(MyParam1(1), MyParam2(2), MyParam1(3), MyParam2(3));
        let t2 = Tensor::randn([1, 2, 3, 3], (Kind::Float, Device::Cpu));
        let my_tensor2 = MyTensor2::new(t2).unwrap();
        assert_eq!(my_tensor2.tensor().size(), &[1, 2, 3, 3]);
    }

    #[test]
    fn test_types() {
        parameter_type!(MyParam1, i64);
        parameter_type!(MyParam2, i64);
        tensor_type!(MyTensor, [MyParam1, MyParam2, MyParam1], Kind::Float);
        MyTensor::set(MyParam1(1), MyParam2(2), MyParam1(3));
        let t = Tensor::randn([1, 2, 3], (Kind::Float, Device::Cpu));
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
        tensor_type!(MyTensor, [MyParam1, MyParam2, MyParam1], Kind::Float);
        MyTensor::set(MyParam1(1), MyParam2(2), MyParam1(3));
        let t = Tensor::randn([1, 2, 3], (Kind::Float, Device::Cpu));
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
    fn test_kind() {
        // Initialize a tensor_type _without_ `tch::` in the kind.
        tensor_type!(MyTensor, [i64, i64], Kind::Int);
        MyTensor::set(1, 3);
        let tensor = Tensor::from_slice(&[0, 1, 2]).reshape([1, 3]);
        let my_tensor = MyTensor::new(tensor).unwrap();
        assert_eq!(*my_tensor.size(), vec![1, 3]);

        // Initialize a tensor_type _with_ `tch::` in the kind.
        tensor_type!(MyTensor2, [i64, i64], tch::Kind::Float);
        MyTensor2::set(1, 3);
        let tensor2 = Tensor::from_slice(&[0, 1, 2]).reshape([1, 3]);
        let my_tensor2 = MyTensor::new(tensor2).unwrap();
        assert_eq!(*my_tensor2.size(), vec![1, 3]);

        // Create with wrong kind.
        let tensor3 = Tensor::from_slice(&[0.0, 1.0, 2.0]); // Double.

        // MyTensor is Int.
        match MyTensor::new(tensor3) {
            Err(MyTensorError::KindMismatch {
                type_name,
                expected,
                found,
            }) => {
                if type_name != "MyTensor" || expected != Kind::Int || found != Kind::Double {
                    panic!("expected KindMismatch, but unexpected type_name ({}), found ({:?}) or expected ({:?})", type_name, found, expected)
                }
            }
            _ => panic!("expected KindMismatch"),
        };
    }

    #[test]
    fn test_clone() {
        tensor_type!(MyTensor, [i64, i64], Kind::Int);
        MyTensor::set(2, 3);
        let tensor = Tensor::from_slice(&[0, 1, 2, 3, 4, 5]).reshape([2, 3]);
        let my_tensor = MyTensor::new(tensor).unwrap();
        // clone() makes a clone of MyTensor with a shallow_clone of the wrapped tensor.
        let cloned_wrapper = my_tensor.clone();

        assert_eq!(cloned_wrapper.size(), vec![2, 3]);
    }

    #[test]
    fn test_apply() {
        parameter_type!(MyParam1, i64);
        parameter_type!(MyParam2, i64);
        tensor_type!(MyTensor, [MyParam1, MyParam2, MyParam1], Kind::Float);
        MyTensor::set(MyParam1(1), MyParam2(2), MyParam1(3));
        let t = Tensor::randn([1, 2, 3], (Kind::Float, Device::Cpu));
        let my_tensor = MyTensor::new(t).unwrap();

        // Test the apply() method.
        let new_my_tensor = my_tensor.apply(|t| t.fill(1.0)).unwrap();
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
