use tensor_types::{parameter_type, tensor_type};

parameter_type!(BatchSize, i64);
parameter_type!(SequenceLength, i64);
parameter_type!(ModelDimension, i64);
pub struct Params {
    batch_size: BatchSize,
    sequence_length: SequenceLength,
    model_dimension: ModelDimension,
}
tensor_type!(
    BatchSeqTensor,
    [batch_size, sequence_length],
    Params,
    tch::Kind::Float
);
tensor_type!(
    BatchSeqModelTensor,
    [batch_size, sequence_length, model_dimension],
    Params,
    tch::Kind::Float
);

fn transform(in1: &BatchSeqTensor, in2: &BatchSeqModelTensor) {
    in1.tensor();
    in2.tensor();
}

fn main() {
    let params = Params {
        batch_size: BatchSize(1),
        sequence_length: SequenceLength(2),
        model_dimension: ModelDimension(3),
    };

    let t1 = tch::Tensor::from_slice(&[1, 2]).reshape(vec![1, 2]);
    let bst_in = BatchSeqTensor::new(t1, &params).unwrap();
    let t2 = tch::Tensor::from_slice(&[1, 2, 3, 4, 5, 6]).reshape(vec![1, 2, 3]);
    let bsdt_in = BatchSeqModelTensor::new(t2, &params).unwrap();

    // This won't compile because the passed-in types don't match the declared types. They're
    // reversed.
    transform(&bsdt_in, &bst_in);
}
