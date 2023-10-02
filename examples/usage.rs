use tch::Tensor;
use tensor_types::{parameter_type, tensor_type, TensorTypeError};

// Shows the basic operations for creating and using TensorTypes.
fn basic_workflow() -> Result<(), TensorTypeError> {
    // 1. Define the types you'll need.
    parameter_type!(NewType1, i64);
    parameter_type!(NewType2, i64);
    tensor_type!(MyStruct, [NewType1, NewType2, NewType1]);

    // 2. At runtime, define the required dimensions using the typed parameters.
    MyStruct::set(NewType1(1), NewType2(2), NewType1(3))?;

    // 3. When you create a new tch::Tensor or receive one from a tch-rs function, wrap it in your
    // TensorType. It will be checked for the correct size when the TensorType is initialized.
    let t = Tensor::randn([1, 2, 3], (tch::Kind::Float, tch::Device::Cpu));
    let my_instance = MyStruct::new(t)?;

    // 4. Use the wrapped tensor as you normally would.
    // let _tensor0 = my_instance.tensor();
    let _deref_tensor = &*my_instance;

    // Ok to call new() again with right-sized tensor to create another instance.
    let t2 = Tensor::randn([1, 2, 3], (tch::Kind::Float, tch::Device::Cpu));
    assert!(MyStruct::new(t2).is_ok());

    Ok(())
}

// These examples expand the above, but show the errors that can occur.
fn safeties() -> Result<(), TensorTypeError> {
    // 1. Define the types you'll need.
    parameter_type!(NewType1, i64);
    parameter_type!(NewType2, i64);
    tensor_type!(MyStruct, [NewType1, NewType2, NewType1]);

    // An error is returned if new() is called before set().
    let t = Tensor::randn([1, 2, 1], (tch::Kind::Float, tch::Device::Cpu));
    assert!(MyStruct::new(t).is_err());

    // 2. At runtime, define the required dimensions using the typed parameters.
    MyStruct::set(NewType1(1), NewType2(2), NewType1(3))?;

    // 3. Now wrap tensors in your type. They'll be checked for the correct size
    // when initialized.
    let t = Tensor::randn([1, 2, 3], (tch::Kind::Float, tch::Device::Cpu));
    let my_struct = MyStruct::new(t)?;

    // An error is returned if new() is called with wrong-sized tensor.
    let t2 = Tensor::randn([1, 2, 1], (tch::Kind::Float, tch::Device::Cpu));
    assert!(MyStruct::new(t2).is_err());

    // Ok to call new() again with right-sized tensor to create another instance.
    let t2 = Tensor::randn([1, 2, 3], (tch::Kind::Float, tch::Device::Cpu));
    assert!(MyStruct::new(t2).is_ok());

    assert_eq!(my_struct.tensor().size(), &[1, 2, 3]);

    // This line won't compile because it's an error to re-use a TensorType name.
    // tensor_type!(MyStruct, [NewType1, NewType2, NewType1, NewType2]);

    Ok(())
}

// Show how to access the tensor after creating a TensorType.
fn details() -> Result<(), TensorTypeError> {
    // Create a TensorType as before.
    parameter_type!(NewType1, i64);
    parameter_type!(NewType2, i64);
    tensor_type!(MyStruct, [NewType1, NewType2, NewType1]);
    MyStruct::set(NewType1(1), NewType2(2), NewType1(3))?;
    let t = Tensor::randn([1, 2, 3], (tch::Kind::Float, tch::Device::Cpu));
    let my_instance = MyStruct::new(t)?;

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
    // println!("The type tensor is {:?}", type_of(my_instance.into_inner()));

    Ok(())
}

// This example shows how to use your own types as parameters to create TensorTypes.
fn custom_type_usage() -> Result<(), TensorTypeError> {
    pub struct MyStruct(i32);

    // Just make sure it implements `From<MyStruct> for i64`.
    impl From<MyStruct> for i64 {
        fn from(item: MyStruct) -> i64 {
            item.0 as i64
        }
    }

    tensor_type!(MyTensorStruct, [MyStruct]);

    let ms = MyStruct(2);

    MyTensorStruct::set(ms)?;

    let t = Tensor::randn([2], (tch::Kind::Float, tch::Device::Cpu));
    let my_instance = MyTensorStruct::new(t)?;
    assert_eq!(my_instance.tensor().size(), &[2]);
    Ok(())
}

fn main() -> Result<(), TensorTypeError> {
    basic_workflow()?;
    safeties()?;
    details()?;
    custom_type_usage()
}
