
- Feature Name: Adding annotatation field to tir.allocate nodes
- Start Date: 2021-06-01
- RFC PR: https://github.com/apache/tvm-rfcs/pull/23
- GitHub Issue: TBD

# 1. Summary

This RFC proposes to annotation field tir.allocate nodes. These annotations can be used as auxiliary hint to future transformations.

# 2. Motivational usecase : pinned memory 

Currently, TVM relies on dynamic (alloc and free style) allocations in runtime to manage the intermediary memory used by operators and the network. This is sometimes not desirable, especially in microTVM.

The current design of [Unified Static Memory Planner (USMP)](https://github.com/apache/tvm-rfcs/pull/9), enables the user option to provide buffers to place workspace and constant tensors.

```
    tvmc compile my_model.tflite 
    --executor=aot 
    --target=accel,c  
    --with-workspace-buffer= "name=dtcm;target=c;size=1000" # Here the size is more of a hint/guide provided to USMP
    --with-workspace-buffer= "name=sram;target=c,accel"
    --with-parameter-buffer= "name=itcm;target=c;size=5000" # Here the size is more of a hint/guide provided to USMP
    --with-parameter-buffer= "name=flash;target=c,accel"
```

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

Therefore, we'd need a way to represent the association of each of these memories, that the user will pin the buffers (e.g. workspace_buffer_2 to SRAM) to. That infomation should flow closer to allocate nodes in TIR in the compilation passes.

At the IR, we ll need to associate each allocate node with one (or more) memory pools that it can end up, because the scheduling might be satisfied with placing buffers in any of the memory pools in a given set of memory pools. Therefore, the scheduler might want the memory planners to decide which memory pool to use based on finding the allocation that fit.

 There are broadly two requirements here :

P1 : Indicate candidate memory pools (a.k.a. PoolInfo Objects -- for further details see [USMP RFC](https://github.com/apache/tvm-rfcs/pull/9)) that each allocate be associated with

P2 : Indicate the final memory pool the allocate will be pinned on


To serve P2, we are proposing to use the existing tag of the storage_scope with 'global.<pool_name>' in the [USMP RFC](https://github.com/apache/tvm-rfcs/pull/9).

To serve P1, this RFC introduces the addition of the annotations field

# 3. Guide-level explanation

This is not particularly a user-facing feature.

The proposal in this RFC is to add an annotation field to tir.allocate node to be used by compilation passes.


 # 4. Reference-level explanation


To serve P1, we propose to use :

```
class AllocateNode : public StmtNode {
 public:
  /*! \brief The buffer variable. */
  Var buffer_var;
  /*! \brief The type of the buffer. */
  DataType dtype;
  /*! \brief The extents of the buffer. */
  Array<PrimExpr> extents;
  /*! \brief Only allocate buffer when condition is satisfied. */
  PrimExpr condition;
  /*! \brief The body to be executed. */
  Stmt body;
  /*!
   * \brief Additional annotations about the loop.
   *
   *  These annotations can be used as auxiliary hint
   *  to future transformations.
   */
+ Map<String, Objectref> annotations;
```

Here the addition of the annotations field could serve offer hints/guides to future passes (i.e. In Unified Static Memory Planner, we could use "candidate_memory_pools" as the key while the value being Map<String, PoolInfo>.)


# Alternatives

 ## S1. Using AttrStmt to associate additional info :

TIR:
 ```
allocate_node_1 = tir.allocate([157323], "int16", "global")
tir.attr(allocate_node_1, "pinned_memory", "foo_memory,bar_memory")
 ``` 

##  S2. Directly as an allocate node argument :

 ```
class AllocateNode : public StmtNode {
 public:
  /*! \brief The buffer variable. */
  Var buffer_var;
  /*! \brief The type of the buffer. */
  DataType dtype;
  /*! \brief The extents of the buffer. */
  Array<PrimExpr> extents;
  /*! \brief Only allocate buffer when condition is satisfied. */
  PrimExpr condition;
  /*! \brief The body to be executed. */
  Stmt body;
  /*! \brief If the allocate is scoped global, this field indicates
   *  which external memories it could be pinned to as a comma seperated
   *  string.
   */
  String pinned_memory;
 ```
TIR:
 ```
allocate_node_1 = tir.allocate([157323], "int16", "global",  "foo_memory,bar_memory")
 ```

 ## S3. Using additional tags in storage_scope

```
/*! \brief class to represent storage scope */
struct StorageScope {
  /*! \brief The rank of the storage */
  StorageRank rank{StorageRank::kGlobal};
  /*! \brief tag for special purpose memories. */
  Array<String> tags;
```


 ```
  /*!
   * \brief Create storage scope from string
   * \param s The string to be parsed.
   * \return The storage scope.
   */
  static StorageScope Create(const std::string& s) {
    StorageScope r;
    if (s.empty()) {/
      r.rank = StorageRank::kGlobal;
    } else if (s.compare(0, 6, "global") == 0) {
      r.rank = StorageRank::kGlobal;
      r.tags = parseTags(s);
 ```


TIR:

```
allocate_node_1 = tir.allocate([157323], "int16", "global.(foo_memory,bar_memory)")
```


Out of the options, S1 seems the most non-invasive. However, we will need special handlers to obtain the information.

S2 does not change the storage scope (or the 'tags') and adds an additional field to allocates to note this information.

S3 fold the information into storage_scope and utilizes the 'tag' denote the memory. The change to IR, is just to support more tags.

However, in the discussion with community, we felt the need to seperate changes that serves P1 and P2, respectively. In fact, for P2 we could re-use the tag of storage_scope without an IR change. For P1, it seems a good and general change to include annotations for the allocate node.

# 5. Drawbacks

None. Its consistent with rest of the IR design and allows other features to use to pass hints for future transformation in the compiler.










