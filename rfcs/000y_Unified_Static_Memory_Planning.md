    Feature Name: Unified Static Memory Planner
    Start Date: 2021 June 1
    RFC PR: #000y
    GitHub Issue: https://github.com/apache/tvm/issues/8404

# Background

Currently, given a ML model primarily TVM will generate two main artifacts :

* A1 : Description of the sequential execution of operators :
  1. If the "executor" is "graph", this would be a JSON
  2. if the "executor" is "aot", this would be a main function describing call graph of operators
* A2 : library of operators (in the form of runtime.Module)

A1 is generally created out of lowering the "main" relay function and A2 is created lowering fused relay primitive functions → TIR PrimFuncs → C or LLVM artifacts of the operator library.

### Is there some sort of memory planning already being performed ?

Yes, there is.

For A1, the inter-(fused) operator tensors are visible in the "main" relay function. Thus, there exists currently a Relay level pass known as "GraphPlanMemory" that works on the Relay IR to share the space used by tensors which are not live simultaneously and are visible between (fused) operators . Currently, the said pass will use Shared Memory Buffer Object memory planning scheme (See https://blog.tensorflow.org/2020/10/optimizing-tensorflow-lite-runtime.html) to perform the planning.

For A2, the operators are lowered to TIR PrimFuncs. There exist a pass called StorageRewrite that more or less does the same thing as "GraphPlanMemory" but on TIR for the tensors visible within (fused) operators and are not live simultaneously.

# Motivation

For embedded use-cases, its widely accepted that aggressive memory optimizations are vital. Intially we are looking at enable memory planning for embedded use-cases using the AoT executor.

Therefore, there exist two main shortcomings of the current approach :

* The memory used by intermediary tensors within operators are not shared between memory used by inter-operator tensors.

Example TIR :
```
    primfn(placeholder_3: handle, placeholder_4: handle, placeholder_5: handle, T_cast_1: handle) -> ()
      attr = { "global_symbol" :  "fused_nn_conv2d_add_fixed_point_multiply_clip_cast_cast_21" ,  "tir.noalias" : True}
      buffers = {T_cast: Buffer(T_cast_2: Pointer(int16), int16, [ 1 ,  56 ,  56 ,  128 ], []),
      placeholder_2: Buffer(placeholder_6: Pointer(int32), int32, [ 1 ,  1 ,  1 ,  128 ], []),
      placeholder: Buffer(placeholder_7: Pointer(int16), int16, [ 1 ,  56 ,  56 , 128 ], []),
      placeholder_1: Buffer(placeholder_8: Pointer(int16), int16, [ 3 ,  3 ,  128 ,  1 ], [])}

       buffer_map = {placeholder_3: placeholder, placeholder_4: placeholder_1, placeholder_5: placeholder_2, T_cast_1: T_cast} {
       attr [PaddedInput: Pointer(int16)]  "storage_scope" =  "global" ;
       allocate(PaddedInput, int16, [ 430592 ]);
       attr [DepthwiseConv2d: Pointer(int32)]  "storage_scope" =  "global" ;

       allocate(DepthwiseConv2d, int32, [ 401408 ]) {
         for (i1: int32,  0 ,  58 ) {
           for (i2: int32,  0 ,  58 ) {
            for(i3: int32,0,128) {
               PaddedInput[(((i1*7424) + (i2*128)) + i3)] = @tir.if_then_else(((((1<= i1) && (i1 < 57)) && (1<= i2)) && (i2 < 57)), (int16*)placeholder_7[((((i1*7168) + (i2* 128 )) + i3) - 7296)], 0i16, dtype=int16)
             }
```

The above TIR snippet shows that two intra operator buffers PaddedInput, DepthwiseConv2d is not visible to Relay Graph Plan Memory to be shared.

* Assumption of local optimization : performing sharing inside the operator first and sub-subsequently sharing that workspace with inter-operator tensors, would be sub-optimal.

Thus, for the embedded use-cases, we'd need a unified static memory planner that performs memory planning of all tensors holistically to achieve best memory utilization.

# Goals

G1. There would be no TVMBackendAlloc(/Free)Workspace calls generated for tir.allocates that could be evaluated at compile time.

Currently, the TVM codegen and the AoT executor relies on TVMB(A/F)W calls to increment/decrement a pointer of user provided workspace buffer. By the end of this set of work, if the backend uses Unified Static Memory Planning, there should not be TVMB(A/F)W calls rather correct offset in to the user provided buffer should be codegen'd for allocates that could be evaluated at compile time. The dynamically sized allocates will remain untouched, thus will be lowered as usual.

G2. The static memory planning algorithm should be changeable.

There are a variety of memory planning algorithms in discussion with different tradeoffs (See https://discuss.tvm.apache.org/t/discussion-alignment-memory-planning/9730 and https://blog.tensorflow.org/2020/10/optimizing-tensorflow-lite-runtime.html). Depending on the topology and schedules of intermediary buffers, the memory planning algorithm should easily be able to be change able. However, the current design ties the algorithm intimately to the IR constructs – making it harder to modularize / change the algorithm w/o inventing a whole new pass. In reality, the outcome of USMP's algorithm is offsets within a given workspace buffer. Moreover, to produce that it should only need to know the sizes of each tensor and their relative liveness. Therefore, the algorithm interface to USMP should be kept simple to be able to add more algorithms.

G3. Multiple pool support (including constants)

Ideally, the user would expect to provide these buffers in the granularity of the memories they'd want to pin them to. E.g., if there are two RW memories : DRAM and SRAM, the buffers need to be identified and pooled by the compiler. Similiarly, for constant data, we need to have a mechanism to allow user to pin them to appropriate memories and addresses in the IR would simply be offsets into the constant buffer(s) provided by the user

# Guide-level explanation

## U1: Most simple use case

### TVMC


```
tvmc compile my_model.tflite --executor=aot --output-format=mlf --target=c
```

 ### Codegen'd artifacts


```
    `//Codegen'd artifacts in metadata.c (lib0.c)`
    const TVMModel my_model = {
       ...
       .entrypoint = &entrypoint,
    }

    static uint8_t workspace_buffer[WORKSPACE_BUFFER_SIZE];
    static const uint8_t parameters_buffer[PARAMETERS_BUFFER_SIZE] = <compiler_generated_constant_data>;

    static int32_t entrypoint(TVMInputs_my_model* inputs, 
                              TVMOutputs_my_model* outputs,
                               TVMContext* context){
        return my_model_main(inputs.input0, 
                             outputs.output0,
                             &workspace_buffer,
                             parameters_buffer,
                             context.resource_handle);
    }
```
```
// metadata.h

    typedef struct {
       uint8_t* input0;
    }  TVMInputs_my_model;

    typedef struct {
       uint8_t* output0;
    }  TVMOutputs_my_model;
```

### User Application
```

    // The User Application 
        extern  const TVMModel my_model;
           int main(...) {
                ...
                TVMInputs_my_model inputs = {my_data};
                TVMOutputs_my_model outputs = {output_space};
                TVMExecute(&my_model,
                           &inputs,
                           &outputs,  
                           NULL);
            }
```
## U2: User wants to share workspaces

### TVMC
```
    tvmc compile my_model_1.tflite
    --executor=aot 
    --output-format=mlf
    --target=accel,c  
    --with-workspace-buffer= "name=sram;target=c,accel"

    tvmc compile my_model_2.tflite 
    --executor=aot
    --output-format=mlf 
    --target=accel,c
    --with-workspace-buffer= "name=sram;target=c,accel"
```
### Codegen'd Artifacts
```
    //Codegen'd artifacts in metadata.c (lib0.c)
    const TVMModel my_model_1 = {
       ...
       .entrypoint = &entrypoint,
    }

    static const uint8_t parameters_buffer[PARAMETERS_BUFFER_SIZE] = <compiler_generated_constant_data>;

     static int32_t entrypoint(TVMInputs_my_model_1* inputs, 
                               TVMOutputs_my_model_1* outputs, 
                               TVMContext* context){
        return my_model_1_main(inputs.input0,
                               outputs.output0,
                               parameters_buffer,
                               context.workspaces.sram, 
                               context.resource_handle);
    }
```
```
// metadata.h

    #define TVM_MY_MODEL_1_SRAM_WORKSPACE_BUFFER_SIZE xxxx

    typedef struct {
       uint8_t* sram;
    }  TVMWorkspaces_my_model_1;

    typedef struct {
       uint8_t* input0;
    }  TVMInputs_my_model_1;

    typedef struct {
       uint8_t* output0;
    }  TVMOutputs_my_model_1;

`//Codegen'd artifacts in metadata.c (lib0.c)`

    const TVMModel my_model_2 = {
       ...
       .entrypoint = &entrypoint,
    }
```
```
    static const uint8_t parameters_buffer[PARAMETERS_BUFFER_SIZE] = <compiler_generated_constant_data>;

    static int32_t entrypoint(TVMInputs_my_model_2* inputs, 
                              TVMOutputs_my_model_2* outputs, 
                              TVMContext* context){
        return my_model_2_main(inputs.input0,
        outputs.output0,
                              parameters_buffer,
                              context.workspaces.sram, 
                              context.resource_handle);
    }
```
```
// metadata.h

    #define TVM_MY_MODEL_2_SRAM_WORKSPACE_BUFFER_SIZE xxxx

    typedef struct {
       uint8_t* sram;
    }  TVMWorkspaces_my_model_2;

    typedef struct {
       uint8_t* input0;
    }  TVMInputs_my_model_2;

    typedef struct {
       uint8_t* output0;
    }  TVMOutputs_my_model_2;
```
### User Application
```
    // The User Application    
        extern  const TVMModel my_model_1;
        extern  const TVMModel my_model_2;

        // Please calculate the maximum of TVM_MY_MODEL_1_SRAM_WORKSPACE_BUFFER_SIZE and TVM_MY_MODEL_2_SRAM_WORKSPACE_BUFFER_SIZE and define it as TVM_MY_MODELS_COMMON_WORKSPACE_BUFFER_SIZE
        // Alternatively, user could use a malloc (if permitted and desired) for runtime calculation of the max
        static uint8_t workspace_buffer[TVM_MY_MODELS_COMMON_WORKSPACE_BUFFER_SIZE];

            int main(...) {
                ...
                TVMContext context;
                TVMInputs_my_model_1 inputs = {my_data_1};
                TVMOutputs_my_model_1 outputs = {output_space_1};
                TVMWorkspaces_my_model_1 workspaces1 = {
                    .sram = &workspace_buffer,
                };
                TVMSetWorkspaces(&context, &workspaces1);
                TVMExecute(&my_model_1, &inputs_1, &outputs_1, &context);
                ...
                TVMInputs_my_model_2 inputs = {my_data_2};
                TVMOutputs_my_model_2 outputs = {output_space_2};
                TVMWorkspaces_my_model_2 workspaces2 = {
                    .sram = &workspace_buffer,
                };
                TVMSetWorkspaces(&context, &workspaces2);
                TVMExecute(&my_model_2, &inputs_2, &outputs_2, &context);
                ...
            }
```
## U3 : User wants to pin buffers to different memories

### TVMC
```
    tvmc compile my_model.tflite 
    --executor=aot 
    --target=accel,c  
    --with-workspace-buffer= "name=dtcm;target=c;size=1000" # Here the size is more of a hint/guide provided to USMP
    --with-workspace-buffer= "name=sram;target=c,accel"
    --with-parameter-buffer= "name=itcm;target=c;size=5000" # Here the size is more of a hint/guide provided to USMP
    --with-parameter-buffer= "name=flash;target=c,accel"
```
### Codegen'd Artifacts
```
    //Codegen'd artifacts in metadata.c (lib0.c)
    const TVMModel my_model = {
       ...
       .entrypoint = &entrypoint,
    }

    static int32_t entrypoint(TVMInputs_my_model* inputs, 
                               TVMOutputs_my_model* outputs, 
                               TVMContext* context){

         return my_model_main(inputs.input0,
                              outputs.output0,
                              context.workspaces.dtcm,
                              context.workspaces.sram,
                              context.parameters.itcm,
                              context.parameters.flash, 
                              context.resource_handle);
    }
```
```
// metadata.h

    #define TVM_MY_MODEL_DTCM_WORKSPACE_BUFFER_SIZE xxxx
    #define TVM_MY_MODEL_SRAM_WORKSPACE_BUFFER_SIZE xxxx
    #define TVM_MY_MODEL_ITCM_PARAMETER_BUFFER_SIZE xxxx
    #define TVM_MY_MODEL_FLASH_PARAMETER_BUFFER_SIZE xxxx

    typedef struct {
       uint8_t* dtcm;
       uint8_t* sram;
    }  TVMWorkspaces_my_model;

    typedef struct {
       uint8_t* itcm;
       uint8_t* flash;
    }  TVMParameters_my_model;

    typedef struct {
       uint8_t* input0;
    }  TVMInputs_my_model;

    typedef struct {
       uint8_t* output0;
    }  TVMOutputs_my_model;
```
### User Application
```
    // The User Application 
        extern  const TVMModel my_model;
        __attribute__((section( "ITCM" )  const uint8_t   my_model_params_1[TVM_MY_MODEL_ITCM_PARAMETER_BUFFER_SIZE] = <param_1_data>;
        __attribute__((section( "FLASH" ), aligned( 16 )))  const uint8_t my_model_params_2[TVM_MY_MODEL_FLASH_PARAMETER_BUFFER_SIZE] = <param_2_data>;
        __attribute__((section( "DTCM" )  static uint8_t workspace_buffer_1[TVM_MY_MODEL_DTCM_WORKSPACE_BUFFER_SIZE];
        __attribute__((section( "SRAM" ), aligned( 16 )))  static uint8_t workspace_buffer_2[TVM_MY_MODEL_SRAM_WORKSPACE_BUFFER_SIZE];

    int main(...) {
         ...
         TVMContext context;
         TVMInputs_my_model_1 inputs = {input};
         TVMOutputs_my_model_1 outputs = {output};
         TVMWorkspaces_my_model workspaces = {
             .sram = &workspace_buffer_1,
             .dtcm = &workspace_buffer_2,
         };
         TVMParameters_my_model parameters = {
             .flash = &my_model_params_1,
             .itcm = &my_model_params_2
         };
         TVMSetWorkspaces(&context, &workspaces);
         TVMSetParameters(&context, parameters);
         TVMExecute(&my_model, &inputs, &outputs, &context);
    }
```
# Reference-level explanation

## Overview

This should be a IRModule (TIR) → IRModule (TIR) pass.

Inputs : 
* AoT TIR PrimFunc ( the control function describing the call graph to operators)
* All Operator Functions
* the maximum size for each pool We could use "pinned_memory" (see below) to tag buffers with suggested priority order determined by the scheduler.

The idea is USMP will try to pool them using the preferred "pinned_memory" and fallback whenever the size is exceeding the user provided max size for each pool (if any)

Outputs : 
* AoT TIR PrimFunc accepting pool buffers from the user.
* All Operator functions accepting pool buffers.
  * Each operator function should address using the correct offset in the correct pool buffer

Special Parametric Inputs : 
* function : The algorithm to be used for planning From a component PoV, the algorithm is a special input with a defined interface.

The current proposal for the interface is as follows :
```
    struct BufferInfo {
        Integer uid;
        Integer size_bytes;
        Integer alignment;
        Array<BufferInfo> conflicts; //the conflicting bufferinfo objs
        Array<Integer> pool_candidates;`
        String pool_name;`
        Integer pool_offset;`
    }
```
```
void (*foo)(Array<ByfferInfo> buffers, Map<String, Integer> pool_sizes)
```
### Special Considerations :

* tir.constants : TIR does not have the ability to represent constants – which is limiting and often leads to having side-channels to carry constants between TIR compiler passes including this one.
Therefore, in this work as a pre-requisite we should aim to fix this by supporting tir.constants (similiar to relay.constants).
  * Why do we need constants expressed in TIR ?
    * If not, it should be represented as inputs to TIR main function (logic : anything that is not expressible in TIR will become inputs). In which case, we would need to associate that Var with a special tag to indicate its constant and its metadata (e.g., desired pools, alignment requirements, etc.)
* Currently "with" or "let" scopes are tree structured and carry transitive property. E.g, if tensor A is live with tensor B && tensor B is live with tensor C → tensor A is live with tensor C – which may not be true always.
Thus current "let" or "with" scopes are unable to express liveness information. Therefore, we'd need a side-channel to express this information.

### How the input TIR to USMP should be lowered ?

##### Step 1 : The bound relay.const in Relay IR should be lowered via TE → TIR as tir.constants
After Step 1 (introducing tir.constants to hold constant data) : the TIR code should like as follows :
```
# This snippet shows the format of pre-USMP pseudo TIR code.

    def main(input1: ty.handle, output1: ty.handle):
       my_model_fused_op1 = tir.allocate(..., pinned_memory=["dtcm", "sram"])
       my_model_fused_op2 = tir.allocate(..., pinned_memory=["sram])
       tir.call("my_model_fused_op1", input1, my_model_fused_op1, fused_op1_weights, fused_op1_biases)
       tir.call( "my_model_fused_op2" , my_model_fused_op1, my_model_fused_op2, fused_op2_weights, fused_op2_biases)

    def my_model_fused_op1(input : ty.handle, output : ty.handle):
       tir.func_attr({"global_symbol":"my_model_fused_op1","tir.noalias": True})
       intermediate_tensor_1 = tir.allocate(..., pinned_memory=["dtcm", "sram"]) # By  default they will have all possible memories
       intermediate_tensor_2 = tir.allocate(..., pinned_memory=["dtcm", "sram"]) # unless scheduler removes them
       weights = tir.allocate_const(..., pinned_memory=["itcm", "flash"])
       biases = tir.allocate_const(..., pinned_memory=["itcm", "flash"])
       ...
       <compute>
       ...

    def my_model_fused_op2(input : ty.handle, output : ty.handle):
       tir.func_attr({"global_symbol":"my_model_fused_op2", "tir.noalias": True})
       intermediate_tensor_1 = tir.allocate(..., pinned_memory=[1, 2])
       intermediate_tensor_2 = tir.allocate(..., pinned_memory=[1, 2])
       weights = tir.allocate_const(..., pinned_memory=["itcm", "flash"])
       biases = tir.allocate_const(..., pinned_memory=["itcm", "flash"])
       ...
       <compute>
       ...
```
##### Step 2 : Run an analysis pass to populate a Map<tir::Var, BufferInfo> that contains buffer information as defined above (See the struct BufferInfo).

##### Step 3 : Use the updated Map<tir::Var, BufferInfo> to generate Array<BufferInfo>, Map<String, Integer> pool_sizes

##### Step 4 : Call the provided/default algorithm (void (*foo)(Array<ByfferInfo> buffers, Map<String, Integer> pool_sizes) to populate pool_id and pool_offset.

##### Step 5 : Use the updated Map<tir::Var, BufferInfo> (with pool_id and pool_offset) mutate the IR that would result as following :
```
# This snippet shows the format of post-USMP pseudo TIR code.

    def main(input1: ty.handle, output1: ty.handle, params_1 : ty.handle, params_2 : ty.handle, workspace_1 : ty.handle, workspace_2 : ty.handle):
       tir.call("my_model_fused_op1", input1, params1, params2, workspace_1, workspace_2)
       tir.call("my_model_fused_op2", params1, params2, workspace_1, workspace_2)

    def my_model_fused_op1(input, params_1, params_2, workspace_1, workspace_2):
       tir.func_attr({"global_symbol":"my_model_fused_op1","tir.noalias":True})
       intermediate_tensor_1=tir.load("uint8", workspace_1.data, <offset>)
       intermediate_tensor_2=tir.load("uint8", workspace_1.data, <offset>)
       output=tir.load("uint8", workspace_1.data, <offset>)
       weights=tir.load("uint8", params_1.data, <offset>)
       biases=tir.load("uint8", params_1.data, <offset>)
       ...
       <compute>
       ...

    def my_model_fused_op2(params_1, params_2, workspace_1, workspace_2):
       tir.func_attr({"global_symbol":"my_model_fused_op2","tir.noalias":True})
       input=tir.load("uint8", workspace_1.data, <offset>)
       intermediate_tensor_1=tir.load("uint8", workspace_1.data, <offset>)
       intermediate_tensor_2=tir.load("uint8", workspace_2.data, <offset>)
       output=tir.load("uint8", workspace_2.data, <offset>)
       weights=tir.load("uint8", params_1.data, <offset>)
       biases=tir.load("uint8", params_2.data, <offset>)
       ...
       <compute>
       ...
```
# Code Structure

* src/tir/usmp/analysis/ -- this is where analysis pases of USMP will live
* src/tir/usmp/transforms/ -- this is where transform pases of USMP will live
* src/tir/usmp/usmp.cc -- this is main intergration of USMP that exposes the full TIR --> TIR transformation as described.
* tests/python/unittest/test_tir_usmp_*.py -- this where unittests for each of the passes and pass pipeline for USMP as a component will live.

NOTE 1: All the above passes will have a mirror in the python.

NOTE 2: to support tir.constants generally, we'll be enhancing the bound relay.constants to be lowered down to tir.constants to codegen. Those changes will appear through out the stack accordingly.