# tensor_types

The tensor_types crate provides strong typing and size-checking to the tensors in
PyTorch Rust frameworks, preventing a large class of hard-to-find bugs. 

### The Problem
PyTorch is a powerful machine-learning library. Rust ports such as
[tch-rs](https://github.com/LaurentMazare/tch-rs) and
[Candle](https://github.com/huggingface/candle) combine its well-designed
architecture to the correctness and reliability of Rust. As its primary data
structure, Torch uses Tensors, a flexible data representation that includes a
rich collection of supporting functions. 

The problem that arises, however, in writing reliable code is that Tensors
become overused and undifferentiated as they are reused throughout a system to
represent different data. For example, in a typical machine learning workflow,
Tensors may represent: tokens in a raw data, embedded tokens, batched sequences
of tokens, batched logits, sequences of probabilities, and finally output
tokens.

Writing code that passes Tensors throughout increases defect risk in three primary
ways:
- Tensors can change shape, possibly propagating incompatible shapes to other
  code. As tensors are transformed and manipulated, it may not be obvious where
  their shapes are modified. Unexpected changes in shape can lead to bugs that
  are hard to find. The Rust compiler can offer no help in this case because the
  representation remains a Tensor no matter the shape.
- Tensors can change kinds. For example, a transformer architecture may accept
  sequences of tokens, a 2d tensor of Int64, but embed it into a 3d tensor of
  Floats. Torch Tensor represent both of these very different types as Tensors.
- Tensors as arguments to functions or structs can be misordered or misassigned.
  You might have meant for your function to operate on `input1` and `input2`,
  but as both are represented by the tch:Tensor type, the compiler can't help
  you catch a function call in which the parameter order is reversed.
  
### The Solution

Making it easy to wrap Tensors into size-checked, kind-checked types increases
reliability, decreases defects, and increases readability. It allows the
compiler to catch bugs that can be hard to find in code that compiles, but
produces wrong results. The `tensor_types` crate allows code like the following:

```rust
    // Define TokenizedInput as a type holding a 2d tensor for batched token 
    // sequences.
    tensor_type!(TokenizedInput, [BatchSize, SequenceLength], Kind::Int64);
    // Define EmbeddedInput as a type holding a 3d tensor for batched sequence
    // of embedded.
    tensor_type!(EmbeddedInput, [BatchSize, SequenceLength, ModelDim], Kind::Float);
    
    let input = tokenizer(...); // Tokenize the input into a tch::Tensor.
    let tokenized = TokenizedInput::new(input)?; // Wrap, checking shape.
    // Embed, confirming the new shape and kind. Here, embed() accepts
    // TokenizedInput and returns a tch::Tensor.
    let embedded = EmbeddedInput::new(embed(tokenized)?)?; 
    // More effectively, you would define embed() to accept a TokenizedInput
    // and return an EmbeddedInput, so the your code would read like:
    // let embedded = embed(tokenized)?; 
```

### Motivation and Examples

As another example, here's a line of buggy code:

```rust
    transform(encoder_input, decoder_input)?;
```
Did you spot the bug?

The definition of `transform()` is:

```rust
    pub fn transform(decoder_input: Tensor, encoder_input: Tensor) -> Result<()> {
```

The buggy line of code passed the arguments in the wrong order, but the compiler
can offer no help because both are Tensors. 


```rust
    // Define EncoderInput as a type holding a 3d tensor.
    tensor_type!(EncoderInput, [BatchSize, SequenceLength, ModelDim], Kind::Float);
    // Define DecoderInput as a type holding a 3d tensor.
    tensor_type!(DecoderInput, [BatchSize, SequenceLength, ModelDim], Kind::Float);
    
    ...
    transform(encoder_input, decoder_input)?; // Won't compile 
```
...where
```rust
    pub fn transform(decoder_input: DecoderInput, encoder_input: EncoderInput) -> Result<()> {
```

Additionally, the TensorTypes define the required shapes of the tensors,
preventing hard-to-find bugs in which tensors change shape unexpectedly. Such
shape changes can occur anywhere in a program as operations are applied to the
tensors.

For example, the following code may have a bug:
```rust
    let input: tch::Tensor =...;
    let output: tch::Tensor = my_function(input)?; // This function may transpose the input.
```
...where
```rust
    pub fn my_function(input: tch::Tensor) -> Result<tch::Tensor> {
```

Did the transpose occur or not? The compiler can't tell. There's no runtime
error if a transpose does or does not happen because whether transposed or not,
the output is a Tensor. Only specifically checking the shape of the output
tensor can tell.

The tensor_types crate makes it easy to maintain the correct tensor shapes as
operations are performed on them. For example:
```rust
    tensor_type!(BatchSeqModel, [BatchSize, SequenceLength, ModelDim], Kind::Float);
    tensor_type!(BatchModelSeq, [BatchSize, ModelDim, SequenceLength], Kind::Float);

    let input: tch::BatchSeqModel =...;
    let output: tch::BatchModelSeq = my_function(input)?; // This function will transpose the input.
```
...where
```rust
    pub fn my_function(input: BatchSeqModel) -> Result<BatchModelSeq> {
        ...
        let output: tch::Tensor = // output from some tch::Tensor operations.
        BatchModelSeq::new(output)
    }
```

Now `my_function()` is clearly defined as returning a transposed result. It
won't compile until the developer makes the code return a transposed shape in a
`BatchModelSeq` type. And at runtime, `my_function` will return a
`ShapeMismatch` if the output tensor does not match the expected shape when
wrapped in `BatchModelSeq::new()` for return from the function.



### Key Features

The key features are:
- Strongly typed tensors: Tensors have a static shape known at compile time.
- Type safe operations: Operations on Tensors are type, size, and kind checked.
- Dimension checking: Operations are checked to have matching dimensions.

### Readability
As the previous examples show, the code is just as readable using Tensortypes as
before, but now includes size and kind checks as it runs. Readability is further
increased as you use TensorTypes throughout your code. Function signatures that
provide no help on their effects on Tensor sizes and kinds like this:
```rust
    fn prepare_input(t: Tensor) -> Result<Tensor, Error> {
        ...
```
...become much more readable like this:
```rust
    fn prepare_input(t: BatchSeq) -> Result<BatchSeqEmbed, Error> {
        ...
```

### Example Usage:
```rust
    // Define your TensorTypes at the start of the program for reuse throughout.
    // Or as needed in each function.
    // 1. Define DecoderInputType as a type holding a 3d tensor. The types that
    //    will give the dimensions for the tensor are given by BatchSize, 
    //    SequenceLength, and ModelDim.
    tensor_type!(DecoderInputType, [BatchSize, SequenceLength, ModelDim], Kind::Float);
    //    Define BatchSeqType as a 2d tensor of tokens, so Int64.
    tensor_type!(BatchSeqType, [SequenceLength, ModelDim], Kind::int64);

    // 2. At runtime, set the required dimensions for the typed parameters.
    //    Because the tensor dimensions are themselves types, these can't be
    //    mixed up.
    DecoderInputType::set(BatchSize(1), SequenceLength(100), ModelDim(256));
    BatchModelType::set(SequenceLength(100), ModelDim(256));

    // 3. Use your new type's new() function to create a new instance of your
    //    type that wraps any tch::Tensor. The tensor will be checked for the
    //    correct size.
    // For example, suppose we obtain t0 from some other function...
    let t0 = Tensor::randn([1, 100, 256], (tch::Kind::Float, tch::Device::Cpu));
    // Wrap it in the DecoderInputType, which will check the size and fail if it
    // is not [BatchSize, SequenceLength, ModelDim], ie, [1, 100, 256].
    let wrapped_t0 = DecoderInputType::new(t0)?;

    // Apply tensor functions. The result is size checked again.
    let new_my_tensor = tokenized_input.apply(|t| t.triu(0))?; // Type: BatchSeqType

    // Or use the tensor in the TensorType directly. No size checking though.
    let cos = *new_my_tensor.cos();  // Type: tch::Tensor
    
    // After a sequence of tch::Tensor operations, you can convert back to a 
    // TensorType to confirm the expected shape.
    let cos = DecoderInputType::new(cos)?;

    // Suppose you have a decoder that will convert from 3d Float to 2d Int64.
    let tokens = my_tokenizer::decode(cos);
    
    BatchModelType::new(tokens)?; // Type: BatchModelType
    ...
```

### Learn More

To learn how to use the tensor_types crate, see
- `examples/usage.rs`: Various ways to use TensorTypes.
- `examples/before_after.rs`: Simple example of errors and how TensorTypes prevents them.
- `tests/*`: Tests illustrate correct usage of TensorTypes.
- `test/compilation_tests/*`: Shows the compilation errors TensorTypes catches.

