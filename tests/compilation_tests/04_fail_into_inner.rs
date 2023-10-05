use tensor_types::tensor_type;

pub struct Params {
    my_param1: i64,
    my_param2: i64,
}
tensor_type!(MyTensor, [my_param1, my_param2], Params, tch::Kind::Double);

fn main() {
    let params = Params {
        my_param1: 2,
        my_param2: 2,
    };

    let tensor = tch::Tensor::from_slice(&[1, 2, 3, 4]).reshape(vec![2, 2]);
    let my_tensor = MyTensor::new(tensor, &params).unwrap();
    println!("{:?}", &my_tensor);
    let _inner = &my_tensor.into_inner();
    // This line fails to compile because `into_inner()` returns ownership of the inner tensor.
    println!("{:?}", &my_tensor);
}
