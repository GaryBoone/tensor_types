use tensor_types::tensor_type;

tensor_type!(MyTensor, [i64, i64]);

fn main() {
    MyTensor::set(2, 2).unwrap();
    let tensor = tch::Tensor::from_slice(&[1, 2, 3, 4]).reshape(vec![2, 2]);
    let my_tensor = MyTensor::new(tensor).unwrap();
    println!("{:?}", &my_tensor);
    let _inner = &my_tensor.into_inner();
    // This line fails to compile because `into_inner()` returns ownership of the inner tensor.
    println!("{:?}", &my_tensor);
}
