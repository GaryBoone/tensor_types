use anyhow::Error;
use tch::{Device, Kind, Tensor};
use tensor_types::{parameter_type, tensor_type, TensorType, TensorTypeError};

// Shows the basic operations for creating and using TensorTypes.
fn basic_workflow() -> Result<(), Error> {
    // 1. Define the types you'll need.
    // parameter_types are used to define the dimensions of the tensor. They're just any type that
    // implements From<i64>.
    parameter_type!(NewType1, i64);
    parameter_type!(NewType2, i64);

    // The tensor types will need a runtime source of the parameter values. Define a structure to
    // hold them.
    pub struct Params {
        new_type_1: NewType1,
        new_type_2: NewType2,
    }

    // 2. Now define the tensor type. The first argument is the name of the type. The second is a
    // list of the parameter fields used to check the wrapped tensor's shape. The third is the type
    // that holds the parameter values. The fourth is the kind of the tensor.
    //
    // So this macro call will define a new type called MyStruct that wraps a tch::Tensor with shape
    // [params.new_type1, params.new_type2, params.new_type1], where params is an instance of
    // Params.
    tensor_type!(
        MyStruct,
        [new_type_1, new_type_2, new_type_1], // Ok to repeat a type.
        Params,
        Kind::Float
    );

    // 3. At runtime, define the required dimensions by filling in the Params structure.
    let params = Params {
        new_type_1: NewType1(1),
        new_type_2: NewType2(2),
    };

    // 4. When you create a new tch::Tensor or receive one from a tch-rs function, wrap it in your
    // TensorType. It will be checked for the correct size by new().
    let t = Tensor::randn([1, 2, 1], (Kind::Float, Device::Cpu));
    let my_instance = MyStruct::new(t, &params)?;

    // 5. Use the wrapped tensor as you normally would.
    // let _tensor0 = my_instance.tensor();
    let _deref_tensor = &*my_instance;

    // You can call new() again with a correctly-sized tensor to create another instance.
    let t2 = Tensor::randn([1, 2, 1], (Kind::Float, Device::Cpu));
    assert!(MyStruct::new(t2, &params).is_ok());

    Ok(())
}

// These examples expand the above, but show the errors that can occur.
fn safeties() -> Result<(), Error> {
    // 1. Define the types you'll need.
    parameter_type!(NewType1, i64);
    parameter_type!(NewType2, i64);
    parameter_type!(NewType3, i64);
    pub struct Params {
        new_type_1: NewType1,
        new_type_2: NewType2,
        new_type_3: NewType3,
    }

    tensor_type!(
        MyStruct,
        [new_type_1, new_type_2, new_type_3],
        Params,
        Kind::Float
    );

    // 2. At runtime, define the required dimensions using the typed parameters.
    let params = Params {
        new_type_1: NewType1(1),
        new_type_2: NewType2(2),
        new_type_3: NewType3(3),
    };

    // 3. Now wrap tensors in your type. They'll be checked for the correct size
    // when initialized.
    let t = Tensor::randn([1, 2, 3], (Kind::Float, Device::Cpu));
    let my_struct = MyStruct::new(t, &params)?;

    // An error is returned if new() is called with wrong-sized tensor.
    let t2 = Tensor::randn([1, 2, 1], (Kind::Float, Device::Cpu));
    assert!(MyStruct::new(t2, &params).is_err());

    // An error is returned if new() is called with wrong kind of tensor.
    let t2 = Tensor::from_slice(&[1, 2, 3, 4, 5, 6])
        .reshape([1, 2, 3])
        .to_kind(Kind::Int16);
    assert!(MyStruct::new(t2, &params).is_err());

    // Ok to call new() again with right-sized tensor to create another instance.
    let t2 = Tensor::randn([1, 2, 3], (Kind::Float, Device::Cpu));
    assert!(MyStruct::new(t2, &params).is_ok());

    assert_eq!(my_struct.tensor().size(), &[1, 2, 3]);

    // This line won't compile because it's an error to re-use a TensorType name.
    // tensor_type!(MyStruct, [new_type_1, new_type_2], Params, Kind::Float);

    Ok(())
}

// Show how to access the tensor after creating a TensorType.
fn details() -> Result<(), Error> {
    parameter_type!(NewType1, i64);
    parameter_type!(NewType2, i64);
    parameter_type!(NewType3, i64);
    pub struct Params {
        new_type_1: NewType1,
        new_type_2: NewType2,
        new_type_3: NewType3,
    }
    tensor_type!(
        MyStruct,
        [new_type_1, new_type_2, new_type_3],
        Params,
        Kind::Float
    );
    let params = Params {
        new_type_1: NewType1(1),
        new_type_2: NewType2(2),
        new_type_3: NewType3(3),
    };

    let t = Tensor::randn([1, 2, 3], (Kind::Float, Device::Cpu));
    let my_instance = MyStruct::new(t, &params)?;

    fn type_of<T>(_: T) -> &'static str {
        std::any::type_name::<T>()
    }

    // Prints: The type tensor is "&usage::details::MyStruct"
    println!("The type tensor is {:?}", type_of(&my_instance));

    // There are several ways to access the tensor.
    // You can obtain the tensor by dereferencing the instance with `*my_instance`. To print it,
    // Prints: The type tensor is "&tch::wrappers::tensor::Tensor"
    println!("The type tensor is {:?}", type_of(my_instance.tensor()));

    // You can obtain the tensor by dereferencing the instance with `*my_instance`. To print it,
    // we have to use `&` because `*my_instance` which is a tch::Tensor doesn't implement Copy.
    // Prints: The type tensor is "&tch::wrappers::tensor::Tensor"
    println!("The type tensor is {:?}", type_of(&*my_instance));

    // Unwrap the tensor from the instance.
    // Prints: The type tensor is "tch::wrappers::tensor::Tensor"
    println!("The type tensor is {:?}", type_of(my_instance.into_inner()));
    // This line won't compile because the tensor has been moved out of the instance.
    // println!("The type tensor is {:?}", type_of(my_instance));

    Ok(())
}

// This example shows how to use your own types as parameters to create TensorTypes.
fn custom_type_usage() -> Result<(), Error> {
    #[derive(Copy, Clone)]
    pub struct MyStruct(i32);

    // Just make sure it implements `From<MyStruct> for i64`.
    impl From<MyStruct> for i64 {
        fn from(item: MyStruct) -> i64 {
            item.0 as i64
        }
    }

    // Rest as before.
    pub struct Params {
        my_struct: MyStruct,
    }

    tensor_type!(MyTensorStruct, [my_struct], Params, Kind::Float);

    let params = Params {
        my_struct: MyStruct(2),
    };

    let t = Tensor::randn([2], (Kind::Float, Device::Cpu));
    let my_instance = MyTensorStruct::new(t, &params)?;
    assert_eq!(my_instance.tensor().size(), &[2]);
    Ok(())
}

// This example exercises the errors that can occur when using TensorTypes.
fn error_handling() -> Result<(), Error> {
    parameter_type!(MyParameterType, i64);
    pub struct Params {
        my_type: MyParameterType,
    }
    tensor_type!(MyStruct, [my_type], Params, Kind::Float);
    let params = Params {
        my_type: MyParameterType(1),
    };

    parameter_type!(Mytype, i64);
    tensor_type!(MyTensorStruct, [my_type], Params, Kind::Float);

    // Try to create a TensorType with the right size: success.
    let t = Tensor::randn([1], (Kind::Float, Device::Cpu));
    let my_instance = MyTensorStruct::new(t, &params)?;
    assert_eq!(my_instance.tensor().size(), &[1]);

    // Try to create a TensorType with the wrong size.
    let t = Tensor::randn([2, 3], (Kind::Float, Device::Cpu));
    match MyTensorStruct::new(t, &params) {
        Ok(_) => println!("new() unexpectedly succeeded, but had the wrong size"),
        Err(e) => match e {
            TensorTypeError::ShapeMismatch {
                type_name,
                expected,
                found,
            } => {
                println!(
                    "new() failed as expected with a ShapeMismatch error on type {}: expected {:?}, but found {:?}",
                    type_name, expected, found)
            }
            _ => {
                println!("new() failed unexpectedly with an error other than ShapeMismatch")
            }
        },
    }

    // Try to create a TensorType with the wrong kind.
    let t = Tensor::from_slice(&[1]).to_kind(Kind::Int64);
    match MyTensorStruct::new(t, &params) {
        Ok(_) => println!("new() unexpectedly succeeded, but had the wrong kind"),
        Err(e) => match e {
            TensorTypeError::KindMismatch {
                type_name,
                expected,
                found,
            } => {
                println!(
                    "new() failed as expected with a KindMismatch error on type {}: expected {:?}, but found {:?}",
                    type_name, expected, found)
            }
            _ => {
                println!("new() failed unexpectedly with an error other than KindMismatch")
            }
        },
    }

    Ok(())
}

fn main() -> Result<(), Error> {
    basic_workflow()?;
    safeties()?;
    details()?;
    custom_type_usage()?;
    error_handling()
}
