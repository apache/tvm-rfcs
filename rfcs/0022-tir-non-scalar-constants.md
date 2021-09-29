
- Feature Name: tir_non_scalar_constants
- Start Date: 2021-06-01
- RFC PR: https://github.com/apache/tvm-rfcs/pull/22
- GitHub Issue: TBD

# 1. Summary

This RFC proposes how non-scalar constants could be represented in TIR and used by passes in the lowering process.

# 2. Motivation 

Currently, the non-scalar constants are represented in Relay (relay.Constant) to be used by relay passes but not in TIR. Therefore, when performing lowering using TIR passes, we have to maintain a side-channel of tir::Var to constant non-scalar data mapping to perform transformations that could use the knowledge where some of the data are constants.

Few example scenarios as further motivation :

## Weight compression

When lowering for accelerators (See [Arm(R) Ethos(TM)-U NPU](https://github.com/apache/tvm-rfcs/pull/11)), certain operations will need to get tiled to co-optimize performance and memory utilization. Such tiling patterns create slices of weights that need compressing that will end up with varying sizes. Therefore, the knowledge of some tir::Vars refer to constants are critical in the level of TIR to perform this.

## Memory Planning

The TIR program has the ability to express both inter and intra operator memory requirement, post-scheduling as explained further by [Unified Static Memory Planning RFC](https://github.com/apache/tvm-rfcs/pull/9). It would be better if the constants could be embedded to the TIR PrimFunc because the memory for constants becomes visible for the memory planner. Moreover, this allows various [target-dependent lowerings](https://github.com/apache/tvm-rfcs/pull/10), to produce TIR PrimFuncs with target-specific constants in it.

## Winograd Constants

The Winograd transformation (used for fast GEMMs) involves multiplication by a hard-coded constant tensor. This is currently accomplished in TE using a complicated TE compute expression with many nested selects. Being able to directly express a constant tensor here would significantly simplify this code. See https://github.com/apache/tvm/blob/9df2ae8eaa8b394013182a7ad09ac57fe401f80e/python/tvm/topi/utils.py#L320-L350.


# 3. Guide-level explanation

This is not particularly a user-facing feature and this will allow constants to be 'linked' to TIR. Intially, tir.allocate_const nodes will only be created during scheduling when -link-params is included in the Target (e.g. to relay.build and to TVMC).

# 4. Reference-level explanation

The proposal is quite simple and it could be explained as follows :

```
@tvm.script.tir
def myfunc():   
   param = tir.allocate_const([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], "int32", [10])
```

This follows closely the semantics of tir.allocate and the difference being it represent a buffer filled with constants.

There are mainly two ways of constants being created in the lowering :

A1. Linking the params of the model (relay.Constants -- currently, the model params would be in Relay as relay.Constant nodes)

A2. Creation/Mutation of constants in the lowering -- these maybe different to the original constants prior to scheduling the Relay into TIR.

For A1, this should only be done if the target support codegeneration of the constant data (i.e. support --link-params) as part of the operator runtime.Module. Therefore, this is executor independent.

For A2, the lowering for targets that support constant as part of the operators, there can be new (differently sized) constants could be created due to optimizations such as weight compression as required by the target.


### IRNode Definition

```
class AllocateConstNode : public StmtNode {
 public:
  /*! \brief The buffer variable. */
  Var buffer_var;
  /*! \brief The optional data associated to the constant.
      This is mutually exclusive to irmod_storage_idx.
   */
  Optional<NDArray> data;
  /*! \brief If the PrimFunc containing the Stmt is added to IRModule,
       this is an optional index to indicate the index within
       "Constants" attribute, that is a Array<NDArray> of IRModule.
   */
  Optional<Integer> irmod_storage_idx;
  /*! \brief The type of the buffer. */
  DataType dtype;
  /*! \brief The extents of the buffer. */
  Array<PrimExpr> extents;
  /*! \brief The body to be executed. */
  Stmt body;
  /*! \brief The constructor with data. */
}

// The constructor to create a IRNode with constant data
// depending on the type of ObjectRef, it will either
// create AllocateConstNode with irmod_storage_idx or data
AllocateConst(Var buffer_var,
              DataType dtype,
              Array<PrimExpr> extents,
              ObjectRef data_or_idx,
              Stmt body,
              Span span);
```


### Storage of constants

Due to concerns of future expansions of centralized storage of constants and adding alternate methods to parse in constants (other than parsing TVMScript), we have decided to store the constants as an IRModule attribute, *when the PrimFunc is added to the IRModule*. 

This will go as a "Constants" key in the DictAttrs where the value is a Array\<NDArray>. However, they are only meant to be accessed via tir.allocate_const(...) nodes in TIR.


* If the constants are created in within passes, the IRModule::Add(...) for a PrimFunc needs to traverse the Stmts to pick the NDArray, add it "Constants" IRModule attribute (Array\<NDArray>) and populate *irmod_storage_idx*.

* If the constants are present in IRModule prior to the PrimFunc is created, then the ObjectRef (for NDArray) and the index of constants in "Constants" IRModule attribute (Array\<NDArray>) has to be populated.




# 5. Drawbacks

* Not all targets need/benefit from handling codegeneration differently for constants.

    If we have to 'link' constants to TIR all the time, there might need a subsequent pass to pull them out. However, its clearer if we just 'link' constants where the target supports and benefits of having them expressed in TIR.

* The IRModule::Add(...) for TIR PrimFuncs need to traverse the statements to add the constants to IRModule if they are not originally referencing the constants present in the IRModule "Constants" attribute.

# 6. Alternatives and Discussion

## Different way of representations

This is initiated from the discussion on [#8472](https://github.com/apache/tvm/pull/8472).

C1 :
```
@tvm.script.tir
def myfunc():
    tir.attrs({
        "link_params": {"model0": array} 
    })        
   my_param_var = tir.get_link_param("model0")
```
C2 :
```
@tvm.script.tir
def myfunc():
    tir.attrs({
        "link_params": {my_param_var: array} 
    })        
```
C3 :
```
@tvm.script.tir
def myfunc():   
   param = tir.allocate_const([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], "int32", [10])
```

C1 and C2 does not need an addition of IR node, however, needs special handling in the passes to figure out whether its a constant.

C3 adds a new IR node, but seems straight-forward way to represent constants near to the compute.

## Different IR node names

D1 : tir.constant
D2 : tir.allocate_const

D1 matches more with relay.Constant and D2 shows the similiarity to tir.allocate node, difference being that the data is constant.






