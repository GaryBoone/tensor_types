use tch::{Device, Kind, Tensor};
use tensor_types::{tensor_type, TensorType, TensorTypeError};

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

// Attach the AttentionTensorTrait to one of our types.
impl AttentionTensorTrait for BatchSeqDModelTensor {}

pub struct Params {
    batch_size: i64,
    sequence_length: i64,
    d_model: i64,
    d_reduced: i64,
}

fn attention<T: TensorType<InnerType = Params> + AttentionTensorTrait>(
    query: &T,
    params: &Params,
) -> Result<T, TensorTypeError> {
    // ... do something with the tensors ...
    query.apply_fn(|t| t.triu(1), params)
}

fn main() {
    let params = Params {
        batch_size: 1,
        sequence_length: 2,
        d_model: 3,
        d_reduced: 4,
    };

    let t = Tensor::randn([1, 2, 3], (Kind::Float, Device::Cpu));
    let query = BatchSeqDModelTensor::new(t, &params).unwrap();
    let _ = attention(&query, &params).unwrap();

    let t = Tensor::randn([1, 2, 4], (Kind::Float, Device::Cpu));
    let query = BatchSeqDReducedTensor::new(t, &params).unwrap();
    // This line will fail to compile because BatchSeqDReducedTensor doesn't implement
    // AttentionTensorTrait.
    let _ = attention(&query, &params).unwrap();
}
