
- Feature Name: tir_non_scalar_constants
- Start Date: 2021-06-01
- RFC PR: https://github.com/apache/tvm-rfcs/pull/22
- GitHub Issue: TBD

# 1. Summary

This RFC proposes how non-scalar constants could be represented in TIR and used by passes in the lowering process.

# 2. Motivation 

Currently, the non-scalar constants could be represented in Relay (relay.Constant) to be used by relay passes but not in TIR. Therefore, when performing lowering using TIR passes, we have to maintain a side-channel of tir::Var to constant non-scalar data mapping to perform transformations that could use the knowledge where some of the data are constants.

Few example scenarios as further motivation :

## Weight compression

When lowering for accelerators (E.g. : [Arm(R) Ethos(TM)-U NPU](https://github.com/apache/tvm-rfcs/pull/11)), certain operations will need to get tiled to co-optimize performance and memory utilization. Such tiling patterns create slices of weights that need compressing that will end up with varying sizes. Therefore, the knowledge of some tir::Vars refer to constants are critical in the level of TIR to perform this.

## Memory Planning

The TIR program has the ability to express both inter and intra operator memory requirement, post-scheduling as explained further by [Unified Static Memory Planning RFC](https://github.com/apache/tvm-rfcs/pull/9). It would be better if the constants could be embedded to the TIR PrimFunc. Moreover, this allows various [target-dependent lowerings](https://github.com/apache/tvm-rfcs/pull/10), to produce TIR PrimFuncs with constants in it.

## Winograd Constants

The Winograd transformation (used for fast GEMMs) involves multiplication by a hard-coded constant tensor. This is currently accomplished in TE using a complicated TE compute expression with many nested selects. Being able to directly express a constant tensor here would significantly simplify this code.


# 3. Guide-level explanation

This is not particularly a user-facing feature and this will allow constants to be 'linked' to TIR. Initially, we are planning to use this with gated on '-link-params' argument for relay.build and TVMC.

# 4. Reference-level explanation

The proposal is quite simple and it could be explained as follows :

```
@tvm.script.tir
def myfunc():   
   param = tir.allocate_const([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], "int32", [10])
```

This follows closely the semantics of tir.allocate and the difference being it represent a buffer filled with constants.

There are mainly two ways of constants being created in the lowering :

A1. Linking the params of the model (relay.Constants)

A2. Creation of constants in the lowering.

For A1, this should only be done if the target support codegeneration of the constant data as part of the operators.

For A2, the lowering for targets that support constant as part of the operators, there can be new (differently sized) constants could be created due to optimizations such as weight compression as required by the target.

# 5. Drawbacks

Not all targets need/benefit from handling codegeneration differently for constants.

If we have to 'link' constants to TIR all the time, there might need a subsequent pass to pull them out. However, its clearer if we just 'link' constants where the target supports and benefits of having them expressed in TIR.

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






