use anyhow::Result;
use tch::{Device, Kind, Tensor};
use tensor_types::{parameter_type, tensor_type};

// This example illustrates how tensor_types crate helps you avoid easy-to-make, hard-to-find bugs.
// In this example, we'll create and run a transformer function that takes a batch of encoder inputs
// and a batch of decoder inputs, and returns a batch of transformer outputs. The 'before' version
// uses tch::Tensors directly. The 'after' version shows how to use the tensor_types crate to avoid
// the bugs in the before version. Can you find the bugs in the before version?

// Note that in this example, we're naming our tensor types like `EncoderInput` and `DecoderInput`
// for clarity in the example. In a real program, you would probably use names that describe the
// types and are useful throughout your program, like `BatchTokens` for a 2d tensor of batched
//  tokens, or `BatchSequenceEmbed` for a 3d tensor of a batched sequence of embedded tokens.

// Example function that creates a tch::Tensor of the given shape. In a real program, it would
// create the input tensor for your encoder. Here it just creates a tch::Tensor with random values.
fn make_encoder_input_untyped(
    batch_size: i64,
    sequence_length: i64,
    model_dimension: i64,
) -> Result<Tensor> {
    Ok(Tensor::rand(
        [batch_size, sequence_length, model_dimension],
        (Kind::Float, Device::Cpu),
    ))
}

// Example function that creates a tch::Tensor of the given shape. In a real program, it would
// create the input tensor for your decoder. Here it just creates a tch::Tensor with random values.
fn make_decoder_input_untyped(
    batch_size: i64,
    model_dimension: i64,
    sequence_length: i64,
) -> Result<Tensor> {
    Ok(Tensor::rand(
        [batch_size, sequence_length, model_dimension],
        (Kind::Float, Device::Cpu),
    ))
}

// Here's the example transformer that accepts our inputs and returns the transformer output.
fn my_transformer_untyped(decoder_input: Tensor, encoder_input: Tensor) -> Result<Tensor> {
    // For our demo, we'll just have the transformer add the encoder input and decoder input.
    Ok(encoder_input + decoder_input)
}

// The demo simply creates some inputs and runs the transformer.
fn before(sequence_length: i64, model_dimension: i64, batch_size: i64) -> Result<()> {
    let encoder_input = make_encoder_input_untyped(batch_size, sequence_length, model_dimension)?;
    let decoder_input = make_decoder_input_untyped(batch_size, sequence_length, model_dimension)?;

    let _out = my_transformer_untyped(encoder_input, decoder_input)?;

    Ok(())
}

// Ok, we've already written several bugs in the code above. Did you spot them? Next, by rewriting
// the above to use TensorTypes, we'll let the compiler find these bugs for us.

// We'll use the tensor_types crate to create the same functions. First, we define our types. The
// parameter types specify our model parameters. We define the types here. Their actual values are
// set in at runtime because they may derive from user input or a configuration file.
parameter_type!(BatchSize, i64);
parameter_type!(SequenceLength, i64);
parameter_type!(ModelDimension, i64);

// Here we define a TensorType. It's just a wrapper for the tch::Tensor, but will be size checked
// with the given ParameterTypes. The EncoderInput will be a a TensorType that wraps a tch::Tensor
// with shape [BatchSize, SequenceLength, ModelDimension].
// For clarity in this demo, the TensorTypes are named as *Input or *Output, but in a real program
// where the TensorTypes would be used throughout the program, it would be clearer to name them
// after the data their dimensions represent, e.g. BatchSequenceEmbed or BatchToken.
// The tensor_type! macro accepts the new type name, followed by the parameter types that define the
// tensor's shape.
tensor_type!(
    EncoderInput,
    [BatchSize, SequenceLength, ModelDimension],
    Kind::Float
);
// The DecoderInput will be a TensorType with shape [BatchSize, SequenceLength, ModelDimension].
tensor_type!(
    DecoderInput,
    [BatchSize, SequenceLength, ModelDimension],
    Kind::Float
);
// The Transformer will output the sequence of tokens, so the TransformerOutput will be a TensorType
// with shape [BatchSize, SequenceLength].
tensor_type!(TransformerOutput, [BatchSize, SequenceLength], Kind::Float);

// The modified transformer function now accepts typed tensors, so it's impossible to pass in the
// wrong tensor for any argument. The compiler will catch such errors for us.
fn my_transformer(
    decoder_input: DecoderInput,
    encoder_input: EncoderInput,
) -> Result<TransformerOutput> {
    // Here, as before, out transformer function will just add the inputs. We can do this operation
    // using the tensor_types apply() function, which will apply a given function to the wrapped
    // tensor, and ensure that the result is still the expected size.
    let sum = decoder_input.apply(|t| t + encoder_input.tensor())?;

    // The transformer function returns a TransformerOutput. So the result must match the size
    // expected by the TransformerOutput. For this demo, we'll just drop the second dimension of the
    // tensor. Here, we illustrate a series of tch::Tensor operations performed without TensorTypes,
    // then wrapped into the return value.
    let cos = sum.tensor().cos(); // `cos` is type tch::Tensor
    let narrowed = cos.narrow(2, 0, 1); // `narrowed` is type tch::Tensor
    let squeezed = narrowed.squeeze(); // `squeezed` is type tch::Tensor

    // We've now dropped a dimension of the tensor. The tch::Tensor representation is flexible and
    // also represents the differently-shaped tensor. But it provides no type safety which can make
    // it hard to find where dimension changes occurred in the code. Instead, here, we'll wrap the
    // tensor in a TransformerOutput, which we've defined above with a specific shape that will be
    // checked by the new() function.
    let transformer_out = TransformerOutput::new(squeezed)?;
    Ok(transformer_out)
}

// The make_encoder_input_typed() function, as before, just creates a tch::Tensor with random
// values. But this one accepts and returns typed tensors. The compiler will catch any errors in the
// argument order or return value shape.
fn make_encoder_input_typed(
    batch_size: BatchSize,
    sequence_length: SequenceLength,
    model_dimension: ModelDimension,
) -> Result<EncoderInput> {
    let t = Tensor::rand(
        [*batch_size, *sequence_length, *model_dimension],
        (Kind::Float, Device::Cpu),
    );
    Ok(EncoderInput::new(t)?)
}

fn make_decoder_input_typed(
    batch_size: BatchSize,
    sequence_length: SequenceLength,
    model_dimension: ModelDimension,
) -> Result<DecoderInput> {
    let t = Tensor::rand(
        [*batch_size, *sequence_length, *model_dimension],
        (Kind::Float, Device::Cpu),
    );
    Ok(DecoderInput::new(t)?)
}

// As before, the demo simply creates some inputs and runs the transformer. Note that unlike before,
// all of the parameters are typed, so it's impossible to pass in the wrong parameter or tensor for
// any argument.
fn after(
    batch_size: BatchSize,
    sequence_length: SequenceLength,
    model_dimension: ModelDimension,
) -> Result<()> {
    let encoder_input = make_encoder_input_typed(batch_size, sequence_length, model_dimension)?;
    let decoder_input = make_decoder_input_typed(batch_size, sequence_length, model_dimension)?;

    let _ = my_transformer(decoder_input, encoder_input)?;
    Ok(())
}

fn main() -> Result<()> {
    // Set up the before version:
    let batch_size = 20;
    // Changing this value will surface a bug in the before version, but only at runtime.
    let sequence_length = 20;
    let model_dimension = 40;

    before(batch_size, sequence_length, model_dimension)?;

    // Set up the after version:
    // The model parameters are now typed. Changing any of these values will cause no errors in the
    // after version because the compiler has prevented any shape mismatch bugs.
    let batch_size = BatchSize(20);
    let sequence_length = SequenceLength(20);
    let model_dimension = ModelDimension(40);

    // Now we use the runtime values to set the shapes of our TensorTypes. Once set, the TensorTypes
    // can be used repeatedly, but their shapes must always match the values set here.
    EncoderInput::set(batch_size, sequence_length, model_dimension);
    DecoderInput::set(batch_size, sequence_length, model_dimension);
    TransformerOutput::set(batch_size, sequence_length);

    after(batch_size, sequence_length, model_dimension)?;
    Ok(())
}
