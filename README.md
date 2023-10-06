# tensor_types

The tensor_types crate provides strong typing and size-checking to the tensors in
PyTorch Rust frameworks, preventing a large class of hard-to-find bugs. 

## Introduction

### The Problem
PyTorch is a powerful machine-learning library. Rust ports such as
[tch-rs](https://github.com/LaurentMazare/tch-rs) and
[Candle](https://github.com/huggingface/candle) combine its well-designed
architecture to the correctness and reliability of Rust. As its primary data
structure, Torch uses Tensors, a flexible data representation that includes a
rich collection of supporting functions. 

A problem that arises, however, in writing reliable tensor-based software is
that Tensors become overused and undifferentiated as they are reused throughout
a system. They're used to represent different data in different parts of the
program, but all with the same data structure. For example, in a machine
learning workflow, Tensors may represent: tokens in a raw data, embedded tokens,
batched sequences of tokens, batched logits, sequences of probabilities, and
finally output tokens.

Writing code that passes Tensors throughout increases defect risk in three primary
ways:
- Tensors can change shape, possibly propagating incompatible shapes to other
  code. As tensors are transformed and manipulated, it may not be obvious where
  their shapes are modified. Unexpected changes in shape can lead to bugs that
  are hard to find. The Rust compiler can offer no help in this case because the
  representation remains a Tensor no matter the shape.
- Tensors can change kinds. For example, a transformer architecture may accept
  sequences of tokens, a 2d tensor of Int64, but embed it into a 3d tensor of
  Floats. Torch Tensors represent both of these very different types as Tensors.
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
    // sequences. TokenizedInput tensors will be checked to be size 
    // [params.batch_size, params.sequence_length] where params is an instance 
    // of Params.
    tensor_type!(TokenizedInput, [batch_size, sequence_length], Params, Kind::Int64);
    // Define EmbeddedInput as a type holding a 3d tensor for batched sequence
    // of embedded.
    tensor_type!(EmbeddedInput, [batch_size, sequence_length, model_dim], Params, Kind::Float);
    
    let params = load_parameters(); // Or however Params is initialized.
    let input = tokenizer(...); // Tokenize the input into a tch::Tensor.
    let tokenized = TokenizedInput::new(input, &params)?; // Wrap, checking shape.
    // Embed, confirming the new shape and kind. Here, embed() accepts
    // TokenizedInput and returns a tch::Tensor.
    let embedded = EmbeddedInput::new(embed(tokenized)?, , &params)?; 
    // More effectively, you would define embed() to accept a TokenizedInput
    // and return an EmbeddedInput, so the your code would read like:
    // let embedded = embed(tokenized, &params)?; 
```

The tensor types created by the `tensor_type!` macro check their sizes against a
structure you define that contains your runtime values. This approach allows
parameters to be loaded once at runtime, perhaps from a configuration file, or
easily set up for testing.

## Motivation and Examples

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
    tensor_type!(EncoderInput, [batch_size, sequence_length, model_dim], Params, Kind::Float);
    // Define DecoderInput as a type holding a 3d tensor.
    tensor_type!(DecoderInput, [batch_size, sequence_length, model_dim], Params, Kind::Float);
    
    ...
    transform(encoder_input, decoder_input)?; // Won't compile. They're backwards. 
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
tensor can tell, assuming the two dimensions are different.

The tensor_types crate makes it easy to maintain the correct tensor shapes as
operations are performed on them. For example:
```rust
    tensor_type!(BatchSeqModel, [batch_size, sequence_length, model_dim], Params, Kind::Float);
    tensor_type!(BatchModelSeq, [batch_size, model_dim, sequence_length], Params, Kind::Float);

    let input: tch::BatchSeqModel =...;
    // This function will transpose the input or return an error if the expected
    // shape change doesn't happen.
    let output: tch::BatchModelSeq = my_function(input)?; 
```
...where
```rust
    pub fn my_function(input: BatchSeqModel) -> Result<BatchModelSeq> {
        ...
        let output: tch::Tensor = // Output from some tch::Tensor operations.
        BatchModelSeq::new(output)
    }
```

Now `my_function()` is clearly defined as returning a transposed result. It
won't compile until the developer makes the code return a transposed shape in a
`BatchModelSeq` type. And at runtime, `my_function` will return a
`ShapeMismatch` if the output tensor does not match the expected shape when
wrapped in `BatchModelSeq::new()` for return from the function.


## Details

### Key Features

The key features are:
- Strongly typed tensors: Tensors have a static shape known at compile time.
- Type safe operations: Operations on Tensors are type, size, and kind checked.
- Dimension checking: Operations are checked to have matching dimensions.

### Readability
As the previous examples show, the code is just as readable using the tensor
types as before, but now includes size and kind checks as it runs. Readability
is further increased as you use TensorTypes throughout your code. Function
signatures that provide no help on their effects on Tensor sizes and kinds like
this:
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
    // 1. Define DecoderInputType as a type holding a 3d tensor. The fields in a 
    //    Params instance that will give the dimensions for the tensor are 
    //    batch_size, sequence_length, and model_dim.
    tensor_type!(DecoderInputType, [batch_size, sequence_length, model_dim], Params, Kind::Float);
    //    Define BatchSeqType as a 2d tensor of tokens, so Int64.
    tensor_type!(BatchSeqType, [sequence_length, model_dim], Params, Kind::int64);

    // Define Params.
    pub struct Params {
        batch_size: i64,
        sequence_length: i64,
        model_dim: i64
    }

    // 2. At runtime, set the required dimensions for the typed parameters.
    let params = Params {
        batch_size: 1, 
        sequence_length: 100, 
        model_dim: 250};

    // 3. Use your new type's new() function to create a new instance of your
    //    type that wraps any tch::Tensor. The tensor will be checked for the
    //    correct size.
    
    // For example, suppose we obtain t0 from some other function...
    let t0 = Tensor::randn([1, 100, 256], (tch::Kind::Float, tch::Device::Cpu));
    // Wrap it in the DecoderInputType, which will check the size and fail if it
    // is not [BatchSize, SequenceLength, ModelDim], ie, [1, 100, 256].
    let wrapped_t0 = DecoderInputType::new(t0, &params)?;

    // Apply tensor functions. The result is size checked again.
    let new_my_tensor = tokenized_input.apply_fn(|t| t.triu(0), &params)?; // Type: BatchSeqType

    // Or use the tensor in the TensorType directly. No size checking though.
    let cos = *new_my_tensor.cos();  // Type: tch::Tensor
    
    // After a sequence of tch::Tensor operations, you can convert back to a 
    // TensorType to confirm the expected shape.
    let cos = DecoderInputType::new(cos, &params)?;

    // Suppose you have a decoder that will convert from 3d Float to 2d Int64.
    let tokens = my_tokenizer::decode_tensor(*cos);  // Type: tch::Tensor

    // Convert into a tensor_type before returning it to validate it.
    BatchModelType::new(tokens, &params)?; // Type: BatchModelType
    ...
```

## Extending the Type
It's easy to add functionality to the types created with the `tensor_types!` macro. 
For example, here is an example type extended to to include directly adding two TensorTypes.
```rust
// BatchSeqDModelTensor: Embedding converts each token to a vector of size
// d_model. They are embedded in an floating point space, so are now kind Float.
tensor_type!(
    BatchSeqDModelTensor,
    [batch_size, sequence_length, d_model],
    ModelParams,
    Kind::Float
);
impl BatchSeqDModelTensor {
    pub fn add(&self, t2: &Self, params: &crate::ModelParams) -> Result<Self> {
        use tensor_types::TensorType;
        Ok(Self::new(&self.tensor + &t2.tensor, params)?)
    }
}
```
`BatchSeqDModelTensor`s can now be added like:
```rust
    pub fn forward_t(
        &self,
        decoder_input: &BatchSeqDModelTensor,
        ...
    ) -> Result<BatchSeqDModelTensor> {
        let masked_mha_output: BatchSeqDModelTensor = ...
        let sum = decoder_input.add(&masked_mha_output, &self.params)?;
```


## Traits and Marker Traits

The types created with the `tensor_type!` macro all implement a trait called
`TensorType`. This trait makes enables Rust trait operations such as polymorphic
arrays and function arguments. Wait! Doesn't that exactly defeat the purpose of
the `tensor_types` crate, which is to make different types unique? Well, yes, if
used directly. Instead, the purpose of the trait is to allow some limited
polymorphism where appropriate. 

For example, perhaps you've added a Local Attention layer that reduces the
dimensionality of your embedded training examples. Now you want the next layer,
a Dense Attention layer, to operate on either the reduced dimension
`BatchSeqDReducedTensor` examples or the full dimension `BatchSeqDModelTensor`
examples. We need a function that can accept either of these, but we don't want
to allow any tensor type or any tch::Tensor. Doing so would effectively remove
size checking.

What we can do is use Rust's Trait Bounds to limit the allowed `TensorTypes`
passed into the function. It's easy. First, define a marker trait and attach it
to the types.

```rust
// AttentionTensorTrait is a marker trait used to limit what can be passed into
// the Attention function.
pub trait AttentionTensorTrait {}

// BatchSeqDReducedTensor are reduced dimensionality tensors produced by the
// Local Attention layer.
tensor_type!(
    BatchSeqDReducedTensor,
    [batch_size, sequence_length, d_reduced],
    ModelParams,
    Kind::Float
);

// Attach the AttentionTensorTrait to our types.
impl AttentionTensorTrait for BatchSeqDModelTensor {}
impl AttentionTensorTrait for BatchSeqDReducedTensor {}
```
Now our function can be defined to only accept these TensorTypes, and not
others.
```rust
    fn attention<T: TensorType<InnerType = Params> + AttentionTensorTrait>(
        query: &T,
        params: &Params,
    ) -> Result<T, TensorTypeError> {
        // Do the attention calculation. [Here, just a tch::Tensor upper 
        // triangle fn, returned directly.]
        query.apply_fn(|t| t.triu(1), params)
    }
```
So our function is defined with a generic argument on TensorType, bringing in
the TensorType methods, and further constrained with the trait bound,
AttentionTensorTrait. Note that `<InnerType = Params>` is how we tell the Rust
compiler about the type we use to provide the tensor type's runtime dimension
values.


## Design and Alternatives Considered

The design of the `tensor_type!` macro was motivated by simplicity and
flexibility. The current version has the format 

```rust
    tensor_type!(<name>, <list of fields>, <struct with those fields>, <kind>);
```

This design requires the parameter instance to be passed throughout the code so
that it can be given to the tensor type's `new()` and other functions that check
the wrapped tensor's dimensions. The parameter instance should be immutable to
ensure consistency in tensor type dimensions over their lifetimes.

This design makes testing easy because test parameter structs can easily be
created and passed into test code as needed. 

### Alternative Design: Fixed Dimensions
An alternative design fixed dimensions as part of the type. That is, the macro
call was like :

```rust
    tensor_type!(<name>, <list of types>, <kind>);
```

The type was created by the macro with the specified dimensions, as it is
currently. However, this version then required a `set()` command to initialize
the runtime values of the dimensions. Once set, the sizes were fixed for the
type. So the setup was like:
```rust
    tensor_type!(DecoderInputType, [BatchSize, SequenceLength, ModelDim], Kind::Float);
    DecoderInputType::set(BatchSize(1), SequenceLength(100), ModelDim(256));
    let my_tensor = DecoderInputType::new(t); // For some tch::Tensor t.
```
This syntax was slightly more concise, and meant that in addition to how many
dimensions were specified, the values of those dimensions were part of the type.
An advantage of this design is that the runtime dimension values do not need to
be passed to the type's `new()` function or other functions that check the
dimensions. 

However, this design is too restrictive.It meant that the tensors needed
internal memory to hold the dimensions given by the `set()` function. It was
implemented with module `static` variables to avoid name collisions and
`std::sync::Once` so that once set, the dimensions were fixed, preventing
changes as is the goal of the `tensor_types` crate. Due to complexity of its
implementation, `proc_macros` were required, increasing the complexity of
testing and packaging by requiring sub-crates. And `Crates.io` doesn't recognize
subcrates, instead treating them as separate crates.

While more concise, the largest disadvantage of this design is that it made
testing quite difficult. In testing, tensor shapes of different sizes are
typically used to exercise a function. For example, a function may be defined
like:

```rust
    pub fn embed(t: BatchTokens) -> Result<BatchTokenEmbed, Error> {
```
`BatchTokens` would be `set()` at program start to be [BatchSize,
SequenceLength] and remain that size throughout the program's lifetime. However,
the tests for `embed()` would need to be run with different shapes. But because
Rust tests are run in parallel, the first test to run would define the shape of
`BatchTokens`. That would cause all other `set()` calls to fail because it can
only be called once. Allowing `set()` to be called repeatedly solved that
problem, but 1) defeated the purpose of using set to fix the dimensions, and 2)
meant that the tensor types had to be protected from mid-test changes due to
thread interleaving as tests ran concurrently. Due to this increased complexity,
this approach was abandoned.

### Alternative Design: Traits
In the current design the `tensor_type!` macro is called with the fields of a
struct that define the runtime values of expected tensor dimensions and the type
of a struct that will provide those fields.

```rust
    tensor_type!(<name>, <list of fields>, <struct with those fields>, <kind>);
```

Another approach considered was to use traits to define the runtime values of
expected tensor dimensions. For example, the macro call might be:
```rust
    tensor_type!(<name>, <list of getters>, <trait with those getters>, <kind>);
```
So an example might be:
```rust
    tensor_type!(BatchSeqType, [get_sequence_length, get_model_dim], ParamsTrait, Kind::int64);

    // Define the Parameters trait.
    pub trait ParamsTrait {
        sequence_length: i64,
        model_dim: i64
    }

    // Define the Params struct.
    pub struct Params {
        sequence_length: i64,
        model_dim: i64
    }

    // Implement the trait for the Params struct.
    impl ParamsTrait for Params {
        fn get_sequence_length(&self) -> i64 {
            self.sequence_length
        }
        fn get_model_dim(&self) -> i64 {
            self.model_dim
        }
    }

    // At runtime, set the required dimensions for the typed parameters.
    let params = Params {
        sequence_length: 100, 
        model_dim: 250};

    let t0 = Tensor::randn([1, 100, 256], (tch::Kind::Float, tch::Device::Cpu));
    let decoder_input = DecoderInputType::new(t0, &params)?;
```
As can be seen, this approach adds quite a bit of boilerplate code to the type
definition just to provide the dimensions. An advantage may be the encapsulation
provided by the trait. However, the macro call to create the new type is
essentially the same as the current design, as are the functions on the new
type. So the advantage of this approach is is outweighed by the burden of
maintaining the trait and the boilerplate code.

### Alternative Design: Coded Dimensions
For completeness, another design considered built the tensor shapes into the
macro code. In this version, no runtime memory is used. A call like:

```rust
    tensor_type!(<name>, value1, value2, value3, ..., <kind>);
```
...can be expanded by the macro system into code essentially like:
```rust
   let expected_size = vec![value1, value2, value3, ...];
   if tensor.size != expected_size {
     return Error...
   }
   ...
``` 
So the code itself stores the values. However, this design also locks the sizes
into the tensor type too much. Specifically, the values of function arguments
must be known at compile time. In addition to eliminating runtime configuration,
it makes testing difficult. For example, as above, once a tensor type is
defined, it can't be changed during testing.

## Learn More

To learn how to use the tensor_types crate, see
- `examples/usage.rs`: Various ways to use TensorTypes.
- `examples/before_after.rs`: Simple example of errors and how TensorTypes prevents them.
- `tests/*`: Tests illustrate correct usage of TensorTypes.
- `test/compilation_tests/*`: Shows the compilation errors TensorTypes catches.

