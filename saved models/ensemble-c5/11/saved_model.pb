??
??
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( ?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
FusedBatchNormV3
x"T

scale"U
offset"U	
mean"U
variance"U
y"T

batch_mean"U
batch_variance"U
reserve_space_1"U
reserve_space_2"U
reserve_space_3"U"
Ttype:
2"
Utype:
2"
epsilonfloat%??8"&
exponential_avg_factorfloat%  ??";
data_formatstringNHWC:
NHWCNCHWNDHWCNCDHW"
is_trainingbool(
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
?
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( ?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ??
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.10.02v2.10.0-rc3-6-g359c3cdfc5f8??
?
Adam/dense_23/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_23/bias/v
y
(Adam/dense_23/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_23/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_23/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_23/kernel/v
?
*Adam/dense_23/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_23/kernel/v*
_output_shapes
:	?*
dtype0
?
Adam/dense_22/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_22/bias/v
z
(Adam/dense_22/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_22/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_22/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_22/kernel/v
?
*Adam/dense_22/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_22/kernel/v* 
_output_shapes
:
??*
dtype0
?
"Adam/batch_normalization_35/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Adam/batch_normalization_35/beta/v
?
6Adam/batch_normalization_35/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_35/beta/v*
_output_shapes	
:?*
dtype0
?
#Adam/batch_normalization_35/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#Adam/batch_normalization_35/gamma/v
?
7Adam/batch_normalization_35/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_35/gamma/v*
_output_shapes	
:?*
dtype0
?
Adam/conv2d_63/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/conv2d_63/bias/v
|
)Adam/conv2d_63/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_63/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/conv2d_63/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*(
shared_nameAdam/conv2d_63/kernel/v
?
+Adam/conv2d_63/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_63/kernel/v*(
_output_shapes
:??*
dtype0
?
"Adam/batch_normalization_34/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Adam/batch_normalization_34/beta/v
?
6Adam/batch_normalization_34/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_34/beta/v*
_output_shapes	
:?*
dtype0
?
#Adam/batch_normalization_34/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#Adam/batch_normalization_34/gamma/v
?
7Adam/batch_normalization_34/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_34/gamma/v*
_output_shapes	
:?*
dtype0
?
Adam/conv2d_62/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/conv2d_62/bias/v
|
)Adam/conv2d_62/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_62/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/conv2d_62/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*(
shared_nameAdam/conv2d_62/kernel/v
?
+Adam/conv2d_62/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_62/kernel/v*(
_output_shapes
:??*
dtype0
?
Adam/conv2d_61/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/conv2d_61/bias/v
|
)Adam/conv2d_61/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_61/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/conv2d_61/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*(
shared_nameAdam/conv2d_61/kernel/v
?
+Adam/conv2d_61/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_61/kernel/v*'
_output_shapes
:@?*
dtype0
?
"Adam/batch_normalization_33/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_33/beta/v
?
6Adam/batch_normalization_33/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_33/beta/v*
_output_shapes
:@*
dtype0
?
#Adam/batch_normalization_33/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/batch_normalization_33/gamma/v
?
7Adam/batch_normalization_33/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_33/gamma/v*
_output_shapes
:@*
dtype0
?
Adam/conv2d_60/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_60/bias/v
{
)Adam/conv2d_60/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_60/bias/v*
_output_shapes
:@*
dtype0
?
Adam/conv2d_60/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv2d_60/kernel/v
?
+Adam/conv2d_60/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_60/kernel/v*&
_output_shapes
:@@*
dtype0
?
Adam/conv2d_59/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_59/bias/v
{
)Adam/conv2d_59/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_59/bias/v*
_output_shapes
:@*
dtype0
?
Adam/conv2d_59/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/conv2d_59/kernel/v
?
+Adam/conv2d_59/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_59/kernel/v*&
_output_shapes
:@*
dtype0
?
Adam/dense_23/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_23/bias/m
y
(Adam/dense_23/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_23/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_23/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_23/kernel/m
?
*Adam/dense_23/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_23/kernel/m*
_output_shapes
:	?*
dtype0
?
Adam/dense_22/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_22/bias/m
z
(Adam/dense_22/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_22/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_22/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_22/kernel/m
?
*Adam/dense_22/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_22/kernel/m* 
_output_shapes
:
??*
dtype0
?
"Adam/batch_normalization_35/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Adam/batch_normalization_35/beta/m
?
6Adam/batch_normalization_35/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_35/beta/m*
_output_shapes	
:?*
dtype0
?
#Adam/batch_normalization_35/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#Adam/batch_normalization_35/gamma/m
?
7Adam/batch_normalization_35/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_35/gamma/m*
_output_shapes	
:?*
dtype0
?
Adam/conv2d_63/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/conv2d_63/bias/m
|
)Adam/conv2d_63/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_63/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/conv2d_63/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*(
shared_nameAdam/conv2d_63/kernel/m
?
+Adam/conv2d_63/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_63/kernel/m*(
_output_shapes
:??*
dtype0
?
"Adam/batch_normalization_34/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Adam/batch_normalization_34/beta/m
?
6Adam/batch_normalization_34/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_34/beta/m*
_output_shapes	
:?*
dtype0
?
#Adam/batch_normalization_34/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#Adam/batch_normalization_34/gamma/m
?
7Adam/batch_normalization_34/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_34/gamma/m*
_output_shapes	
:?*
dtype0
?
Adam/conv2d_62/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/conv2d_62/bias/m
|
)Adam/conv2d_62/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_62/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/conv2d_62/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*(
shared_nameAdam/conv2d_62/kernel/m
?
+Adam/conv2d_62/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_62/kernel/m*(
_output_shapes
:??*
dtype0
?
Adam/conv2d_61/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/conv2d_61/bias/m
|
)Adam/conv2d_61/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_61/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/conv2d_61/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*(
shared_nameAdam/conv2d_61/kernel/m
?
+Adam/conv2d_61/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_61/kernel/m*'
_output_shapes
:@?*
dtype0
?
"Adam/batch_normalization_33/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_33/beta/m
?
6Adam/batch_normalization_33/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_33/beta/m*
_output_shapes
:@*
dtype0
?
#Adam/batch_normalization_33/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/batch_normalization_33/gamma/m
?
7Adam/batch_normalization_33/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_33/gamma/m*
_output_shapes
:@*
dtype0
?
Adam/conv2d_60/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_60/bias/m
{
)Adam/conv2d_60/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_60/bias/m*
_output_shapes
:@*
dtype0
?
Adam/conv2d_60/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv2d_60/kernel/m
?
+Adam/conv2d_60/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_60/kernel/m*&
_output_shapes
:@@*
dtype0
?
Adam/conv2d_59/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_59/bias/m
{
)Adam/conv2d_59/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_59/bias/m*
_output_shapes
:@*
dtype0
?
Adam/conv2d_59/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/conv2d_59/kernel/m
?
+Adam/conv2d_59/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_59/kernel/m*&
_output_shapes
:@*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
r
dense_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_23/bias
k
!dense_23/bias/Read/ReadVariableOpReadVariableOpdense_23/bias*
_output_shapes
:*
dtype0
{
dense_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?* 
shared_namedense_23/kernel
t
#dense_23/kernel/Read/ReadVariableOpReadVariableOpdense_23/kernel*
_output_shapes
:	?*
dtype0
s
dense_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_22/bias
l
!dense_22/bias/Read/ReadVariableOpReadVariableOpdense_22/bias*
_output_shapes	
:?*
dtype0
|
dense_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_22/kernel
u
#dense_22/kernel/Read/ReadVariableOpReadVariableOpdense_22/kernel* 
_output_shapes
:
??*
dtype0
?
&batch_normalization_35/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&batch_normalization_35/moving_variance
?
:batch_normalization_35/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_35/moving_variance*
_output_shapes	
:?*
dtype0
?
"batch_normalization_35/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"batch_normalization_35/moving_mean
?
6batch_normalization_35/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_35/moving_mean*
_output_shapes	
:?*
dtype0
?
batch_normalization_35/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namebatch_normalization_35/beta
?
/batch_normalization_35/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_35/beta*
_output_shapes	
:?*
dtype0
?
batch_normalization_35/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_namebatch_normalization_35/gamma
?
0batch_normalization_35/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_35/gamma*
_output_shapes	
:?*
dtype0
u
conv2d_63/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d_63/bias
n
"conv2d_63/bias/Read/ReadVariableOpReadVariableOpconv2d_63/bias*
_output_shapes	
:?*
dtype0
?
conv2d_63/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*!
shared_nameconv2d_63/kernel

$conv2d_63/kernel/Read/ReadVariableOpReadVariableOpconv2d_63/kernel*(
_output_shapes
:??*
dtype0
?
&batch_normalization_34/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&batch_normalization_34/moving_variance
?
:batch_normalization_34/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_34/moving_variance*
_output_shapes	
:?*
dtype0
?
"batch_normalization_34/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"batch_normalization_34/moving_mean
?
6batch_normalization_34/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_34/moving_mean*
_output_shapes	
:?*
dtype0
?
batch_normalization_34/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namebatch_normalization_34/beta
?
/batch_normalization_34/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_34/beta*
_output_shapes	
:?*
dtype0
?
batch_normalization_34/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_namebatch_normalization_34/gamma
?
0batch_normalization_34/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_34/gamma*
_output_shapes	
:?*
dtype0
u
conv2d_62/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d_62/bias
n
"conv2d_62/bias/Read/ReadVariableOpReadVariableOpconv2d_62/bias*
_output_shapes	
:?*
dtype0
?
conv2d_62/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*!
shared_nameconv2d_62/kernel

$conv2d_62/kernel/Read/ReadVariableOpReadVariableOpconv2d_62/kernel*(
_output_shapes
:??*
dtype0
u
conv2d_61/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d_61/bias
n
"conv2d_61/bias/Read/ReadVariableOpReadVariableOpconv2d_61/bias*
_output_shapes	
:?*
dtype0
?
conv2d_61/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*!
shared_nameconv2d_61/kernel
~
$conv2d_61/kernel/Read/ReadVariableOpReadVariableOpconv2d_61/kernel*'
_output_shapes
:@?*
dtype0
?
&batch_normalization_33/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_33/moving_variance
?
:batch_normalization_33/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_33/moving_variance*
_output_shapes
:@*
dtype0
?
"batch_normalization_33/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_33/moving_mean
?
6batch_normalization_33/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_33/moving_mean*
_output_shapes
:@*
dtype0
?
batch_normalization_33/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_33/beta
?
/batch_normalization_33/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_33/beta*
_output_shapes
:@*
dtype0
?
batch_normalization_33/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_33/gamma
?
0batch_normalization_33/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_33/gamma*
_output_shapes
:@*
dtype0
t
conv2d_60/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_60/bias
m
"conv2d_60/bias/Read/ReadVariableOpReadVariableOpconv2d_60/bias*
_output_shapes
:@*
dtype0
?
conv2d_60/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameconv2d_60/kernel
}
$conv2d_60/kernel/Read/ReadVariableOpReadVariableOpconv2d_60/kernel*&
_output_shapes
:@@*
dtype0
t
conv2d_59/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_59/bias
m
"conv2d_59/bias/Read/ReadVariableOpReadVariableOpconv2d_59/bias*
_output_shapes
:@*
dtype0
?
conv2d_59/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameconv2d_59/kernel
}
$conv2d_59/kernel/Read/ReadVariableOpReadVariableOpconv2d_59/kernel*&
_output_shapes
:@*
dtype0
?
serving_default_conv2d_59_inputPlaceholder*/
_output_shapes
:?????????8*
dtype0*$
shape:?????????8
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_59_inputconv2d_59/kernelconv2d_59/biasconv2d_60/kernelconv2d_60/biasbatch_normalization_33/gammabatch_normalization_33/beta"batch_normalization_33/moving_mean&batch_normalization_33/moving_varianceconv2d_61/kernelconv2d_61/biasconv2d_62/kernelconv2d_62/biasbatch_normalization_34/gammabatch_normalization_34/beta"batch_normalization_34/moving_mean&batch_normalization_34/moving_varianceconv2d_63/kernelconv2d_63/biasbatch_normalization_35/gammabatch_normalization_35/beta"batch_normalization_35/moving_mean&batch_normalization_35/moving_variancedense_22/kerneldense_22/biasdense_23/kerneldense_23/bias*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *.
f)R'
%__inference_signature_wrapper_2445475

NoOpNoOp
??
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*ژ
valueϘB˘ BØ
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer-6
layer_with_weights-5
layer-7
	layer_with_weights-6
	layer-8

layer-9
layer_with_weights-7
layer-10
layer-11
layer_with_weights-8
layer-12
layer_with_weights-9
layer-13
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures*
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
  _jit_compiled_convolution_op*
?
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses

'kernel
(bias
 )_jit_compiled_convolution_op*
?
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses* 
?
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses
6axis
	7gamma
8beta
9moving_mean
:moving_variance*
?
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
*@&call_and_return_all_conditional_losses

Akernel
Bbias
 C_jit_compiled_convolution_op*
?
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
H__call__
*I&call_and_return_all_conditional_losses

Jkernel
Kbias
 L_jit_compiled_convolution_op*
?
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses* 
?
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses
Yaxis
	Zgamma
[beta
\moving_mean
]moving_variance*
?
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses

dkernel
ebias
 f_jit_compiled_convolution_op*
?
g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses* 
?
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
q__call__
*r&call_and_return_all_conditional_losses
saxis
	tgamma
ubeta
vmoving_mean
wmoving_variance*
?
x	variables
ytrainable_variables
zregularization_losses
{	keras_api
|__call__
*}&call_and_return_all_conditional_losses* 
?
~	variables
trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias*
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias*
?
0
1
'2
(3
74
85
96
:7
A8
B9
J10
K11
Z12
[13
\14
]15
d16
e17
t18
u19
v20
w21
?22
?23
?24
?25*
?
0
1
'2
(3
74
85
A6
B7
J8
K9
Z10
[11
d12
e13
t14
u15
?16
?17
?18
?19*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
:
?trace_0
?trace_1
?trace_2
?trace_3* 
:
?trace_0
?trace_1
?trace_2
?trace_3* 
* 
?
	?iter
?beta_1
?beta_2

?decay
?learning_ratem?m?'m?(m?7m?8m?Am?Bm?Jm?Km?Zm?[m?dm?em?tm?um?	?m?	?m?	?m?	?m?v?v?'v?(v?7v?8v?Av?Bv?Jv?Kv?Zv?[v?dv?ev?tv?uv?	?v?	?v?	?v?	?v?*

?serving_default* 

0
1*

0
1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
`Z
VARIABLE_VALUEconv2d_59/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_59/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

'0
(1*

'0
(1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
`Z
VARIABLE_VALUEconv2d_60/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_60/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 
 
70
81
92
:3*

70
81*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses*

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_33/gamma5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_33/beta4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_33/moving_mean;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_33/moving_variance?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*

A0
B1*

A0
B1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
;	variables
<trainable_variables
=regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
`Z
VARIABLE_VALUEconv2d_61/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_61/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

J0
K1*

J0
K1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
D	variables
Etrainable_variables
Fregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
`Z
VARIABLE_VALUEconv2d_62/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_62/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 
 
Z0
[1
\2
]3*

Z0
[1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses*

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_34/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_34/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_34/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_34/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*

d0
e1*

d0
e1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
`Z
VARIABLE_VALUEconv2d_63/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_63/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 
 
t0
u1
v2
w3*

t0
u1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
m	variables
ntrainable_variables
oregularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses*

?trace_0
?trace_1* 

?trace_0
?trace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_35/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_35/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_35/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_35/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
x	variables
ytrainable_variables
zregularization_losses
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses* 

?trace_0* 

?trace_0* 

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
~	variables
trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
_Y
VARIABLE_VALUEdense_22/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_22/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?0
?1*
* 
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses*

?trace_0* 

?trace_0* 
_Y
VARIABLE_VALUEdense_23/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_23/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE*
.
90
:1
\2
]3
v4
w5*
j
0
1
2
3
4
5
6
7
	8

9
10
11
12
13*

?0
?1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

90
:1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

\0
]1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 

v0
w1*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
<
?	variables
?	keras_api

?total

?count*
M
?	variables
?	keras_api

?total

?count
?
_fn_kwargs*

?0
?1*

?	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

?0
?1*

?	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
?}
VARIABLE_VALUEAdam/conv2d_59/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_59/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/conv2d_60/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_60/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_33/gamma/mQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/batch_normalization_33/beta/mPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/conv2d_61/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_61/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/conv2d_62/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_62/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_34/gamma/mQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/batch_normalization_34/beta/mPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/conv2d_63/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_63/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_35/gamma/mQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/batch_normalization_35/beta/mPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEAdam/dense_22/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_22/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEAdam/dense_23/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_23/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/conv2d_59/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_59/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/conv2d_60/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_60/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_33/gamma/vQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/batch_normalization_33/beta/vPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/conv2d_61/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_61/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/conv2d_62/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_62/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_34/gamma/vQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/batch_normalization_34/beta/vPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/conv2d_63/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_63/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_35/gamma/vQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/batch_normalization_35/beta/vPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEAdam/dense_22/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_22/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEAdam/dense_23/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_23/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_59/kernel/Read/ReadVariableOp"conv2d_59/bias/Read/ReadVariableOp$conv2d_60/kernel/Read/ReadVariableOp"conv2d_60/bias/Read/ReadVariableOp0batch_normalization_33/gamma/Read/ReadVariableOp/batch_normalization_33/beta/Read/ReadVariableOp6batch_normalization_33/moving_mean/Read/ReadVariableOp:batch_normalization_33/moving_variance/Read/ReadVariableOp$conv2d_61/kernel/Read/ReadVariableOp"conv2d_61/bias/Read/ReadVariableOp$conv2d_62/kernel/Read/ReadVariableOp"conv2d_62/bias/Read/ReadVariableOp0batch_normalization_34/gamma/Read/ReadVariableOp/batch_normalization_34/beta/Read/ReadVariableOp6batch_normalization_34/moving_mean/Read/ReadVariableOp:batch_normalization_34/moving_variance/Read/ReadVariableOp$conv2d_63/kernel/Read/ReadVariableOp"conv2d_63/bias/Read/ReadVariableOp0batch_normalization_35/gamma/Read/ReadVariableOp/batch_normalization_35/beta/Read/ReadVariableOp6batch_normalization_35/moving_mean/Read/ReadVariableOp:batch_normalization_35/moving_variance/Read/ReadVariableOp#dense_22/kernel/Read/ReadVariableOp!dense_22/bias/Read/ReadVariableOp#dense_23/kernel/Read/ReadVariableOp!dense_23/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp+Adam/conv2d_59/kernel/m/Read/ReadVariableOp)Adam/conv2d_59/bias/m/Read/ReadVariableOp+Adam/conv2d_60/kernel/m/Read/ReadVariableOp)Adam/conv2d_60/bias/m/Read/ReadVariableOp7Adam/batch_normalization_33/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_33/beta/m/Read/ReadVariableOp+Adam/conv2d_61/kernel/m/Read/ReadVariableOp)Adam/conv2d_61/bias/m/Read/ReadVariableOp+Adam/conv2d_62/kernel/m/Read/ReadVariableOp)Adam/conv2d_62/bias/m/Read/ReadVariableOp7Adam/batch_normalization_34/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_34/beta/m/Read/ReadVariableOp+Adam/conv2d_63/kernel/m/Read/ReadVariableOp)Adam/conv2d_63/bias/m/Read/ReadVariableOp7Adam/batch_normalization_35/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_35/beta/m/Read/ReadVariableOp*Adam/dense_22/kernel/m/Read/ReadVariableOp(Adam/dense_22/bias/m/Read/ReadVariableOp*Adam/dense_23/kernel/m/Read/ReadVariableOp(Adam/dense_23/bias/m/Read/ReadVariableOp+Adam/conv2d_59/kernel/v/Read/ReadVariableOp)Adam/conv2d_59/bias/v/Read/ReadVariableOp+Adam/conv2d_60/kernel/v/Read/ReadVariableOp)Adam/conv2d_60/bias/v/Read/ReadVariableOp7Adam/batch_normalization_33/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_33/beta/v/Read/ReadVariableOp+Adam/conv2d_61/kernel/v/Read/ReadVariableOp)Adam/conv2d_61/bias/v/Read/ReadVariableOp+Adam/conv2d_62/kernel/v/Read/ReadVariableOp)Adam/conv2d_62/bias/v/Read/ReadVariableOp7Adam/batch_normalization_34/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_34/beta/v/Read/ReadVariableOp+Adam/conv2d_63/kernel/v/Read/ReadVariableOp)Adam/conv2d_63/bias/v/Read/ReadVariableOp7Adam/batch_normalization_35/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_35/beta/v/Read/ReadVariableOp*Adam/dense_22/kernel/v/Read/ReadVariableOp(Adam/dense_22/bias/v/Read/ReadVariableOp*Adam/dense_23/kernel/v/Read/ReadVariableOp(Adam/dense_23/bias/v/Read/ReadVariableOpConst*X
TinQ
O2M	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__traced_save_2446404
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_59/kernelconv2d_59/biasconv2d_60/kernelconv2d_60/biasbatch_normalization_33/gammabatch_normalization_33/beta"batch_normalization_33/moving_mean&batch_normalization_33/moving_varianceconv2d_61/kernelconv2d_61/biasconv2d_62/kernelconv2d_62/biasbatch_normalization_34/gammabatch_normalization_34/beta"batch_normalization_34/moving_mean&batch_normalization_34/moving_varianceconv2d_63/kernelconv2d_63/biasbatch_normalization_35/gammabatch_normalization_35/beta"batch_normalization_35/moving_mean&batch_normalization_35/moving_variancedense_22/kerneldense_22/biasdense_23/kerneldense_23/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcountAdam/conv2d_59/kernel/mAdam/conv2d_59/bias/mAdam/conv2d_60/kernel/mAdam/conv2d_60/bias/m#Adam/batch_normalization_33/gamma/m"Adam/batch_normalization_33/beta/mAdam/conv2d_61/kernel/mAdam/conv2d_61/bias/mAdam/conv2d_62/kernel/mAdam/conv2d_62/bias/m#Adam/batch_normalization_34/gamma/m"Adam/batch_normalization_34/beta/mAdam/conv2d_63/kernel/mAdam/conv2d_63/bias/m#Adam/batch_normalization_35/gamma/m"Adam/batch_normalization_35/beta/mAdam/dense_22/kernel/mAdam/dense_22/bias/mAdam/dense_23/kernel/mAdam/dense_23/bias/mAdam/conv2d_59/kernel/vAdam/conv2d_59/bias/vAdam/conv2d_60/kernel/vAdam/conv2d_60/bias/v#Adam/batch_normalization_33/gamma/v"Adam/batch_normalization_33/beta/vAdam/conv2d_61/kernel/vAdam/conv2d_61/bias/vAdam/conv2d_62/kernel/vAdam/conv2d_62/bias/v#Adam/batch_normalization_34/gamma/v"Adam/batch_normalization_34/beta/vAdam/conv2d_63/kernel/vAdam/conv2d_63/bias/v#Adam/batch_normalization_35/gamma/v"Adam/batch_normalization_35/beta/vAdam/dense_22/kernel/vAdam/dense_22/bias/vAdam/dense_23/kernel/vAdam/dense_23/bias/v*W
TinP
N2L*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference__traced_restore_2446639??
?
?
S__inference_batch_normalization_35_layer_call_and_return_conditional_losses_2444691

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?

?
E__inference_dense_22_layer_call_and_return_conditional_losses_2446136

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
F__inference_conv2d_59_layer_call_and_return_conditional_losses_2444751

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????6@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????6@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????6@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????6@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????8: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????8
 
_user_specified_nameinputs
?
?
/__inference_sequential_13_layer_call_fn_2445270
conv2d_59_input!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@
	unknown_6:@$
	unknown_7:@?
	unknown_8:	?%
	unknown_9:??

unknown_10:	?

unknown_11:	?

unknown_12:	?

unknown_13:	?

unknown_14:	?&

unknown_15:??

unknown_16:	?

unknown_17:	?

unknown_18:	?

unknown_19:	?

unknown_20:	?

unknown_21:
??

unknown_22:	?

unknown_23:	?

unknown_24:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_59_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_sequential_13_layer_call_and_return_conditional_losses_2445158o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:?????????8: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:?????????8
)
_user_specified_nameconv2d_59_input
?
?
S__inference_batch_normalization_34_layer_call_and_return_conditional_losses_2446013

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_35_layer_call_fn_2446069

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_35_layer_call_and_return_conditional_losses_2444722?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
+__inference_conv2d_62_layer_call_fn_2445930

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_62_layer_call_and_return_conditional_losses_2444812x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????
?: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:?????????
?
 
_user_specified_nameinputs
?
?
+__inference_conv2d_59_layer_call_fn_2445798

inputs!
unknown:@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????6@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_59_layer_call_and_return_conditional_losses_2444751w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????6@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????8: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????8
 
_user_specified_nameinputs
?
?
F__inference_conv2d_63_layer_call_and_return_conditional_losses_2444839

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	?*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	?Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????	?j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????	?w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
i
M__inference_max_pooling2d_34_layer_call_and_return_conditional_losses_2444590

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
i
M__inference_max_pooling2d_33_layer_call_and_return_conditional_losses_2444514

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_33_layer_call_and_return_conditional_losses_2444539

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
N
2__inference_max_pooling2d_33_layer_call_fn_2445834

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_max_pooling2d_33_layer_call_and_return_conditional_losses_2444514?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_33_layer_call_and_return_conditional_losses_2445883

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@?
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
/__inference_sequential_13_layer_call_fn_2445532

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@
	unknown_6:@$
	unknown_7:@?
	unknown_8:	?%
	unknown_9:??

unknown_10:	?

unknown_11:	?

unknown_12:	?

unknown_13:	?

unknown_14:	?&

unknown_15:??

unknown_16:	?

unknown_17:	?

unknown_18:	?

unknown_19:	?

unknown_20:	?

unknown_21:
??

unknown_22:	?

unknown_23:	?

unknown_24:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_sequential_13_layer_call_and_return_conditional_losses_2444898o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:?????????8: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????8
 
_user_specified_nameinputs
?
?
F__inference_conv2d_62_layer_call_and_return_conditional_losses_2444812

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????
?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????
?
 
_user_specified_nameinputs
?H
?
J__inference_sequential_13_layer_call_and_return_conditional_losses_2444898

inputs+
conv2d_59_2444752:@
conv2d_59_2444754:@+
conv2d_60_2444769:@@
conv2d_60_2444771:@,
batch_normalization_33_2444775:@,
batch_normalization_33_2444777:@,
batch_normalization_33_2444779:@,
batch_normalization_33_2444781:@,
conv2d_61_2444796:@? 
conv2d_61_2444798:	?-
conv2d_62_2444813:?? 
conv2d_62_2444815:	?-
batch_normalization_34_2444819:	?-
batch_normalization_34_2444821:	?-
batch_normalization_34_2444823:	?-
batch_normalization_34_2444825:	?-
conv2d_63_2444840:?? 
conv2d_63_2444842:	?-
batch_normalization_35_2444846:	?-
batch_normalization_35_2444848:	?-
batch_normalization_35_2444850:	?-
batch_normalization_35_2444852:	?$
dense_22_2444875:
??
dense_22_2444877:	?#
dense_23_2444892:	?
dense_23_2444894:
identity??.batch_normalization_33/StatefulPartitionedCall?.batch_normalization_34/StatefulPartitionedCall?.batch_normalization_35/StatefulPartitionedCall?!conv2d_59/StatefulPartitionedCall?!conv2d_60/StatefulPartitionedCall?!conv2d_61/StatefulPartitionedCall?!conv2d_62/StatefulPartitionedCall?!conv2d_63/StatefulPartitionedCall? dense_22/StatefulPartitionedCall? dense_23/StatefulPartitionedCall?
!conv2d_59/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_59_2444752conv2d_59_2444754*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????6@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_59_layer_call_and_return_conditional_losses_2444751?
!conv2d_60/StatefulPartitionedCallStatefulPartitionedCall*conv2d_59/StatefulPartitionedCall:output:0conv2d_60_2444769conv2d_60_2444771*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????4@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_60_layer_call_and_return_conditional_losses_2444768?
 max_pooling2d_33/PartitionedCallPartitionedCall*conv2d_60/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_max_pooling2d_33_layer_call_and_return_conditional_losses_2444514?
.batch_normalization_33/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_33/PartitionedCall:output:0batch_normalization_33_2444775batch_normalization_33_2444777batch_normalization_33_2444779batch_normalization_33_2444781*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_33_layer_call_and_return_conditional_losses_2444539?
!conv2d_61/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_33/StatefulPartitionedCall:output:0conv2d_61_2444796conv2d_61_2444798*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????
?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_61_layer_call_and_return_conditional_losses_2444795?
!conv2d_62/StatefulPartitionedCallStatefulPartitionedCall*conv2d_61/StatefulPartitionedCall:output:0conv2d_62_2444813conv2d_62_2444815*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_62_layer_call_and_return_conditional_losses_2444812?
 max_pooling2d_34/PartitionedCallPartitionedCall*conv2d_62/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_max_pooling2d_34_layer_call_and_return_conditional_losses_2444590?
.batch_normalization_34/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_34/PartitionedCall:output:0batch_normalization_34_2444819batch_normalization_34_2444821batch_normalization_34_2444823batch_normalization_34_2444825*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_34_layer_call_and_return_conditional_losses_2444615?
!conv2d_63/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_34/StatefulPartitionedCall:output:0conv2d_63_2444840conv2d_63_2444842*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????	?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_63_layer_call_and_return_conditional_losses_2444839?
 max_pooling2d_35/PartitionedCallPartitionedCall*conv2d_63/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_max_pooling2d_35_layer_call_and_return_conditional_losses_2444666?
.batch_normalization_35/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_35/PartitionedCall:output:0batch_normalization_35_2444846batch_normalization_35_2444848batch_normalization_35_2444850batch_normalization_35_2444852*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_35_layer_call_and_return_conditional_losses_2444691?
flatten_11/PartitionedCallPartitionedCall7batch_normalization_35/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_flatten_11_layer_call_and_return_conditional_losses_2444861?
 dense_22/StatefulPartitionedCallStatefulPartitionedCall#flatten_11/PartitionedCall:output:0dense_22_2444875dense_22_2444877*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_22_layer_call_and_return_conditional_losses_2444874?
 dense_23/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0dense_23_2444892dense_23_2444894*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_23_layer_call_and_return_conditional_losses_2444891x
IdentityIdentity)dense_23/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp/^batch_normalization_33/StatefulPartitionedCall/^batch_normalization_34/StatefulPartitionedCall/^batch_normalization_35/StatefulPartitionedCall"^conv2d_59/StatefulPartitionedCall"^conv2d_60/StatefulPartitionedCall"^conv2d_61/StatefulPartitionedCall"^conv2d_62/StatefulPartitionedCall"^conv2d_63/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:?????????8: : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_33/StatefulPartitionedCall.batch_normalization_33/StatefulPartitionedCall2`
.batch_normalization_34/StatefulPartitionedCall.batch_normalization_34/StatefulPartitionedCall2`
.batch_normalization_35/StatefulPartitionedCall.batch_normalization_35/StatefulPartitionedCall2F
!conv2d_59/StatefulPartitionedCall!conv2d_59/StatefulPartitionedCall2F
!conv2d_60/StatefulPartitionedCall!conv2d_60/StatefulPartitionedCall2F
!conv2d_61/StatefulPartitionedCall!conv2d_61/StatefulPartitionedCall2F
!conv2d_62/StatefulPartitionedCall!conv2d_62/StatefulPartitionedCall2F
!conv2d_63/StatefulPartitionedCall!conv2d_63/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall:W S
/
_output_shapes
:?????????8
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_35_layer_call_and_return_conditional_losses_2444722

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
??
?
"__inference__wrapped_model_2444505
conv2d_59_inputP
6sequential_13_conv2d_59_conv2d_readvariableop_resource:@E
7sequential_13_conv2d_59_biasadd_readvariableop_resource:@P
6sequential_13_conv2d_60_conv2d_readvariableop_resource:@@E
7sequential_13_conv2d_60_biasadd_readvariableop_resource:@J
<sequential_13_batch_normalization_33_readvariableop_resource:@L
>sequential_13_batch_normalization_33_readvariableop_1_resource:@[
Msequential_13_batch_normalization_33_fusedbatchnormv3_readvariableop_resource:@]
Osequential_13_batch_normalization_33_fusedbatchnormv3_readvariableop_1_resource:@Q
6sequential_13_conv2d_61_conv2d_readvariableop_resource:@?F
7sequential_13_conv2d_61_biasadd_readvariableop_resource:	?R
6sequential_13_conv2d_62_conv2d_readvariableop_resource:??F
7sequential_13_conv2d_62_biasadd_readvariableop_resource:	?K
<sequential_13_batch_normalization_34_readvariableop_resource:	?M
>sequential_13_batch_normalization_34_readvariableop_1_resource:	?\
Msequential_13_batch_normalization_34_fusedbatchnormv3_readvariableop_resource:	?^
Osequential_13_batch_normalization_34_fusedbatchnormv3_readvariableop_1_resource:	?R
6sequential_13_conv2d_63_conv2d_readvariableop_resource:??F
7sequential_13_conv2d_63_biasadd_readvariableop_resource:	?K
<sequential_13_batch_normalization_35_readvariableop_resource:	?M
>sequential_13_batch_normalization_35_readvariableop_1_resource:	?\
Msequential_13_batch_normalization_35_fusedbatchnormv3_readvariableop_resource:	?^
Osequential_13_batch_normalization_35_fusedbatchnormv3_readvariableop_1_resource:	?I
5sequential_13_dense_22_matmul_readvariableop_resource:
??E
6sequential_13_dense_22_biasadd_readvariableop_resource:	?H
5sequential_13_dense_23_matmul_readvariableop_resource:	?D
6sequential_13_dense_23_biasadd_readvariableop_resource:
identity??Dsequential_13/batch_normalization_33/FusedBatchNormV3/ReadVariableOp?Fsequential_13/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1?3sequential_13/batch_normalization_33/ReadVariableOp?5sequential_13/batch_normalization_33/ReadVariableOp_1?Dsequential_13/batch_normalization_34/FusedBatchNormV3/ReadVariableOp?Fsequential_13/batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1?3sequential_13/batch_normalization_34/ReadVariableOp?5sequential_13/batch_normalization_34/ReadVariableOp_1?Dsequential_13/batch_normalization_35/FusedBatchNormV3/ReadVariableOp?Fsequential_13/batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1?3sequential_13/batch_normalization_35/ReadVariableOp?5sequential_13/batch_normalization_35/ReadVariableOp_1?.sequential_13/conv2d_59/BiasAdd/ReadVariableOp?-sequential_13/conv2d_59/Conv2D/ReadVariableOp?.sequential_13/conv2d_60/BiasAdd/ReadVariableOp?-sequential_13/conv2d_60/Conv2D/ReadVariableOp?.sequential_13/conv2d_61/BiasAdd/ReadVariableOp?-sequential_13/conv2d_61/Conv2D/ReadVariableOp?.sequential_13/conv2d_62/BiasAdd/ReadVariableOp?-sequential_13/conv2d_62/Conv2D/ReadVariableOp?.sequential_13/conv2d_63/BiasAdd/ReadVariableOp?-sequential_13/conv2d_63/Conv2D/ReadVariableOp?-sequential_13/dense_22/BiasAdd/ReadVariableOp?,sequential_13/dense_22/MatMul/ReadVariableOp?-sequential_13/dense_23/BiasAdd/ReadVariableOp?,sequential_13/dense_23/MatMul/ReadVariableOp?
-sequential_13/conv2d_59/Conv2D/ReadVariableOpReadVariableOp6sequential_13_conv2d_59_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
sequential_13/conv2d_59/Conv2DConv2Dconv2d_59_input5sequential_13/conv2d_59/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????6@*
paddingVALID*
strides
?
.sequential_13/conv2d_59/BiasAdd/ReadVariableOpReadVariableOp7sequential_13_conv2d_59_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
sequential_13/conv2d_59/BiasAddBiasAdd'sequential_13/conv2d_59/Conv2D:output:06sequential_13/conv2d_59/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????6@?
sequential_13/conv2d_59/ReluRelu(sequential_13/conv2d_59/BiasAdd:output:0*
T0*/
_output_shapes
:?????????6@?
-sequential_13/conv2d_60/Conv2D/ReadVariableOpReadVariableOp6sequential_13_conv2d_60_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
sequential_13/conv2d_60/Conv2DConv2D*sequential_13/conv2d_59/Relu:activations:05sequential_13/conv2d_60/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????4@*
paddingVALID*
strides
?
.sequential_13/conv2d_60/BiasAdd/ReadVariableOpReadVariableOp7sequential_13_conv2d_60_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
sequential_13/conv2d_60/BiasAddBiasAdd'sequential_13/conv2d_60/Conv2D:output:06sequential_13/conv2d_60/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????4@?
sequential_13/conv2d_60/ReluRelu(sequential_13/conv2d_60/BiasAdd:output:0*
T0*/
_output_shapes
:?????????4@?
&sequential_13/max_pooling2d_33/MaxPoolMaxPool*sequential_13/conv2d_60/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
?
3sequential_13/batch_normalization_33/ReadVariableOpReadVariableOp<sequential_13_batch_normalization_33_readvariableop_resource*
_output_shapes
:@*
dtype0?
5sequential_13/batch_normalization_33/ReadVariableOp_1ReadVariableOp>sequential_13_batch_normalization_33_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
Dsequential_13/batch_normalization_33/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_13_batch_normalization_33_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
Fsequential_13/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_13_batch_normalization_33_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
5sequential_13/batch_normalization_33/FusedBatchNormV3FusedBatchNormV3/sequential_13/max_pooling2d_33/MaxPool:output:0;sequential_13/batch_normalization_33/ReadVariableOp:value:0=sequential_13/batch_normalization_33/ReadVariableOp_1:value:0Lsequential_13/batch_normalization_33/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_13/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
is_training( ?
-sequential_13/conv2d_61/Conv2D/ReadVariableOpReadVariableOp6sequential_13_conv2d_61_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
sequential_13/conv2d_61/Conv2DConv2D9sequential_13/batch_normalization_33/FusedBatchNormV3:y:05sequential_13/conv2d_61/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????
?*
paddingVALID*
strides
?
.sequential_13/conv2d_61/BiasAdd/ReadVariableOpReadVariableOp7sequential_13_conv2d_61_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
sequential_13/conv2d_61/BiasAddBiasAdd'sequential_13/conv2d_61/Conv2D:output:06sequential_13/conv2d_61/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????
??
sequential_13/conv2d_61/ReluRelu(sequential_13/conv2d_61/BiasAdd:output:0*
T0*0
_output_shapes
:?????????
??
-sequential_13/conv2d_62/Conv2D/ReadVariableOpReadVariableOp6sequential_13_conv2d_62_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
sequential_13/conv2d_62/Conv2DConv2D*sequential_13/conv2d_61/Relu:activations:05sequential_13/conv2d_62/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
?
.sequential_13/conv2d_62/BiasAdd/ReadVariableOpReadVariableOp7sequential_13_conv2d_62_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
sequential_13/conv2d_62/BiasAddBiasAdd'sequential_13/conv2d_62/Conv2D:output:06sequential_13/conv2d_62/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:???????????
sequential_13/conv2d_62/ReluRelu(sequential_13/conv2d_62/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
&sequential_13/max_pooling2d_34/MaxPoolMaxPool*sequential_13/conv2d_62/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
?
3sequential_13/batch_normalization_34/ReadVariableOpReadVariableOp<sequential_13_batch_normalization_34_readvariableop_resource*
_output_shapes	
:?*
dtype0?
5sequential_13/batch_normalization_34/ReadVariableOp_1ReadVariableOp>sequential_13_batch_normalization_34_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
Dsequential_13/batch_normalization_34/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_13_batch_normalization_34_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Fsequential_13/batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_13_batch_normalization_34_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
5sequential_13/batch_normalization_34/FusedBatchNormV3FusedBatchNormV3/sequential_13/max_pooling2d_34/MaxPool:output:0;sequential_13/batch_normalization_34/ReadVariableOp:value:0=sequential_13/batch_normalization_34/ReadVariableOp_1:value:0Lsequential_13/batch_normalization_34/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_13/batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( ?
-sequential_13/conv2d_63/Conv2D/ReadVariableOpReadVariableOp6sequential_13_conv2d_63_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
sequential_13/conv2d_63/Conv2DConv2D9sequential_13/batch_normalization_34/FusedBatchNormV3:y:05sequential_13/conv2d_63/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	?*
paddingVALID*
strides
?
.sequential_13/conv2d_63/BiasAdd/ReadVariableOpReadVariableOp7sequential_13_conv2d_63_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
sequential_13/conv2d_63/BiasAddBiasAdd'sequential_13/conv2d_63/Conv2D:output:06sequential_13/conv2d_63/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	??
sequential_13/conv2d_63/ReluRelu(sequential_13/conv2d_63/BiasAdd:output:0*
T0*0
_output_shapes
:?????????	??
&sequential_13/max_pooling2d_35/MaxPoolMaxPool*sequential_13/conv2d_63/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
?
3sequential_13/batch_normalization_35/ReadVariableOpReadVariableOp<sequential_13_batch_normalization_35_readvariableop_resource*
_output_shapes	
:?*
dtype0?
5sequential_13/batch_normalization_35/ReadVariableOp_1ReadVariableOp>sequential_13_batch_normalization_35_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
Dsequential_13/batch_normalization_35/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_13_batch_normalization_35_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Fsequential_13/batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_13_batch_normalization_35_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
5sequential_13/batch_normalization_35/FusedBatchNormV3FusedBatchNormV3/sequential_13/max_pooling2d_35/MaxPool:output:0;sequential_13/batch_normalization_35/ReadVariableOp:value:0=sequential_13/batch_normalization_35/ReadVariableOp_1:value:0Lsequential_13/batch_normalization_35/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_13/batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( o
sequential_13/flatten_11/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ?
 sequential_13/flatten_11/ReshapeReshape9sequential_13/batch_normalization_35/FusedBatchNormV3:y:0'sequential_13/flatten_11/Const:output:0*
T0*(
_output_shapes
:???????????
,sequential_13/dense_22/MatMul/ReadVariableOpReadVariableOp5sequential_13_dense_22_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
sequential_13/dense_22/MatMulMatMul)sequential_13/flatten_11/Reshape:output:04sequential_13/dense_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
-sequential_13/dense_22/BiasAdd/ReadVariableOpReadVariableOp6sequential_13_dense_22_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
sequential_13/dense_22/BiasAddBiasAdd'sequential_13/dense_22/MatMul:product:05sequential_13/dense_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
sequential_13/dense_22/ReluRelu'sequential_13/dense_22/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
,sequential_13/dense_23/MatMul/ReadVariableOpReadVariableOp5sequential_13_dense_23_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
sequential_13/dense_23/MatMulMatMul)sequential_13/dense_22/Relu:activations:04sequential_13/dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
-sequential_13/dense_23/BiasAdd/ReadVariableOpReadVariableOp6sequential_13_dense_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_13/dense_23/BiasAddBiasAdd'sequential_13/dense_23/MatMul:product:05sequential_13/dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
sequential_13/dense_23/SoftmaxSoftmax'sequential_13/dense_23/BiasAdd:output:0*
T0*'
_output_shapes
:?????????w
IdentityIdentity(sequential_13/dense_23/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOpE^sequential_13/batch_normalization_33/FusedBatchNormV3/ReadVariableOpG^sequential_13/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_14^sequential_13/batch_normalization_33/ReadVariableOp6^sequential_13/batch_normalization_33/ReadVariableOp_1E^sequential_13/batch_normalization_34/FusedBatchNormV3/ReadVariableOpG^sequential_13/batch_normalization_34/FusedBatchNormV3/ReadVariableOp_14^sequential_13/batch_normalization_34/ReadVariableOp6^sequential_13/batch_normalization_34/ReadVariableOp_1E^sequential_13/batch_normalization_35/FusedBatchNormV3/ReadVariableOpG^sequential_13/batch_normalization_35/FusedBatchNormV3/ReadVariableOp_14^sequential_13/batch_normalization_35/ReadVariableOp6^sequential_13/batch_normalization_35/ReadVariableOp_1/^sequential_13/conv2d_59/BiasAdd/ReadVariableOp.^sequential_13/conv2d_59/Conv2D/ReadVariableOp/^sequential_13/conv2d_60/BiasAdd/ReadVariableOp.^sequential_13/conv2d_60/Conv2D/ReadVariableOp/^sequential_13/conv2d_61/BiasAdd/ReadVariableOp.^sequential_13/conv2d_61/Conv2D/ReadVariableOp/^sequential_13/conv2d_62/BiasAdd/ReadVariableOp.^sequential_13/conv2d_62/Conv2D/ReadVariableOp/^sequential_13/conv2d_63/BiasAdd/ReadVariableOp.^sequential_13/conv2d_63/Conv2D/ReadVariableOp.^sequential_13/dense_22/BiasAdd/ReadVariableOp-^sequential_13/dense_22/MatMul/ReadVariableOp.^sequential_13/dense_23/BiasAdd/ReadVariableOp-^sequential_13/dense_23/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:?????????8: : : : : : : : : : : : : : : : : : : : : : : : : : 2?
Dsequential_13/batch_normalization_33/FusedBatchNormV3/ReadVariableOpDsequential_13/batch_normalization_33/FusedBatchNormV3/ReadVariableOp2?
Fsequential_13/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1Fsequential_13/batch_normalization_33/FusedBatchNormV3/ReadVariableOp_12j
3sequential_13/batch_normalization_33/ReadVariableOp3sequential_13/batch_normalization_33/ReadVariableOp2n
5sequential_13/batch_normalization_33/ReadVariableOp_15sequential_13/batch_normalization_33/ReadVariableOp_12?
Dsequential_13/batch_normalization_34/FusedBatchNormV3/ReadVariableOpDsequential_13/batch_normalization_34/FusedBatchNormV3/ReadVariableOp2?
Fsequential_13/batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1Fsequential_13/batch_normalization_34/FusedBatchNormV3/ReadVariableOp_12j
3sequential_13/batch_normalization_34/ReadVariableOp3sequential_13/batch_normalization_34/ReadVariableOp2n
5sequential_13/batch_normalization_34/ReadVariableOp_15sequential_13/batch_normalization_34/ReadVariableOp_12?
Dsequential_13/batch_normalization_35/FusedBatchNormV3/ReadVariableOpDsequential_13/batch_normalization_35/FusedBatchNormV3/ReadVariableOp2?
Fsequential_13/batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1Fsequential_13/batch_normalization_35/FusedBatchNormV3/ReadVariableOp_12j
3sequential_13/batch_normalization_35/ReadVariableOp3sequential_13/batch_normalization_35/ReadVariableOp2n
5sequential_13/batch_normalization_35/ReadVariableOp_15sequential_13/batch_normalization_35/ReadVariableOp_12`
.sequential_13/conv2d_59/BiasAdd/ReadVariableOp.sequential_13/conv2d_59/BiasAdd/ReadVariableOp2^
-sequential_13/conv2d_59/Conv2D/ReadVariableOp-sequential_13/conv2d_59/Conv2D/ReadVariableOp2`
.sequential_13/conv2d_60/BiasAdd/ReadVariableOp.sequential_13/conv2d_60/BiasAdd/ReadVariableOp2^
-sequential_13/conv2d_60/Conv2D/ReadVariableOp-sequential_13/conv2d_60/Conv2D/ReadVariableOp2`
.sequential_13/conv2d_61/BiasAdd/ReadVariableOp.sequential_13/conv2d_61/BiasAdd/ReadVariableOp2^
-sequential_13/conv2d_61/Conv2D/ReadVariableOp-sequential_13/conv2d_61/Conv2D/ReadVariableOp2`
.sequential_13/conv2d_62/BiasAdd/ReadVariableOp.sequential_13/conv2d_62/BiasAdd/ReadVariableOp2^
-sequential_13/conv2d_62/Conv2D/ReadVariableOp-sequential_13/conv2d_62/Conv2D/ReadVariableOp2`
.sequential_13/conv2d_63/BiasAdd/ReadVariableOp.sequential_13/conv2d_63/BiasAdd/ReadVariableOp2^
-sequential_13/conv2d_63/Conv2D/ReadVariableOp-sequential_13/conv2d_63/Conv2D/ReadVariableOp2^
-sequential_13/dense_22/BiasAdd/ReadVariableOp-sequential_13/dense_22/BiasAdd/ReadVariableOp2\
,sequential_13/dense_22/MatMul/ReadVariableOp,sequential_13/dense_22/MatMul/ReadVariableOp2^
-sequential_13/dense_23/BiasAdd/ReadVariableOp-sequential_13/dense_23/BiasAdd/ReadVariableOp2\
,sequential_13/dense_23/MatMul/ReadVariableOp,sequential_13/dense_23/MatMul/ReadVariableOp:` \
/
_output_shapes
:?????????8
)
_user_specified_nameconv2d_59_input
?
?
S__inference_batch_normalization_34_layer_call_and_return_conditional_losses_2445995

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
??
?!
 __inference__traced_save_2446404
file_prefix/
+savev2_conv2d_59_kernel_read_readvariableop-
)savev2_conv2d_59_bias_read_readvariableop/
+savev2_conv2d_60_kernel_read_readvariableop-
)savev2_conv2d_60_bias_read_readvariableop;
7savev2_batch_normalization_33_gamma_read_readvariableop:
6savev2_batch_normalization_33_beta_read_readvariableopA
=savev2_batch_normalization_33_moving_mean_read_readvariableopE
Asavev2_batch_normalization_33_moving_variance_read_readvariableop/
+savev2_conv2d_61_kernel_read_readvariableop-
)savev2_conv2d_61_bias_read_readvariableop/
+savev2_conv2d_62_kernel_read_readvariableop-
)savev2_conv2d_62_bias_read_readvariableop;
7savev2_batch_normalization_34_gamma_read_readvariableop:
6savev2_batch_normalization_34_beta_read_readvariableopA
=savev2_batch_normalization_34_moving_mean_read_readvariableopE
Asavev2_batch_normalization_34_moving_variance_read_readvariableop/
+savev2_conv2d_63_kernel_read_readvariableop-
)savev2_conv2d_63_bias_read_readvariableop;
7savev2_batch_normalization_35_gamma_read_readvariableop:
6savev2_batch_normalization_35_beta_read_readvariableopA
=savev2_batch_normalization_35_moving_mean_read_readvariableopE
Asavev2_batch_normalization_35_moving_variance_read_readvariableop.
*savev2_dense_22_kernel_read_readvariableop,
(savev2_dense_22_bias_read_readvariableop.
*savev2_dense_23_kernel_read_readvariableop,
(savev2_dense_23_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop6
2savev2_adam_conv2d_59_kernel_m_read_readvariableop4
0savev2_adam_conv2d_59_bias_m_read_readvariableop6
2savev2_adam_conv2d_60_kernel_m_read_readvariableop4
0savev2_adam_conv2d_60_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_33_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_33_beta_m_read_readvariableop6
2savev2_adam_conv2d_61_kernel_m_read_readvariableop4
0savev2_adam_conv2d_61_bias_m_read_readvariableop6
2savev2_adam_conv2d_62_kernel_m_read_readvariableop4
0savev2_adam_conv2d_62_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_34_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_34_beta_m_read_readvariableop6
2savev2_adam_conv2d_63_kernel_m_read_readvariableop4
0savev2_adam_conv2d_63_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_35_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_35_beta_m_read_readvariableop5
1savev2_adam_dense_22_kernel_m_read_readvariableop3
/savev2_adam_dense_22_bias_m_read_readvariableop5
1savev2_adam_dense_23_kernel_m_read_readvariableop3
/savev2_adam_dense_23_bias_m_read_readvariableop6
2savev2_adam_conv2d_59_kernel_v_read_readvariableop4
0savev2_adam_conv2d_59_bias_v_read_readvariableop6
2savev2_adam_conv2d_60_kernel_v_read_readvariableop4
0savev2_adam_conv2d_60_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_33_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_33_beta_v_read_readvariableop6
2savev2_adam_conv2d_61_kernel_v_read_readvariableop4
0savev2_adam_conv2d_61_bias_v_read_readvariableop6
2savev2_adam_conv2d_62_kernel_v_read_readvariableop4
0savev2_adam_conv2d_62_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_34_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_34_beta_v_read_readvariableop6
2savev2_adam_conv2d_63_kernel_v_read_readvariableop4
0savev2_adam_conv2d_63_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_35_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_35_beta_v_read_readvariableop5
1savev2_adam_dense_22_kernel_v_read_readvariableop3
/savev2_adam_dense_22_bias_v_read_readvariableop5
1savev2_adam_dense_23_kernel_v_read_readvariableop3
/savev2_adam_dense_23_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : ?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ?)
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:L*
dtype0*?)
value?)B?)LB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:L*
dtype0*?
value?B?LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ? 
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_59_kernel_read_readvariableop)savev2_conv2d_59_bias_read_readvariableop+savev2_conv2d_60_kernel_read_readvariableop)savev2_conv2d_60_bias_read_readvariableop7savev2_batch_normalization_33_gamma_read_readvariableop6savev2_batch_normalization_33_beta_read_readvariableop=savev2_batch_normalization_33_moving_mean_read_readvariableopAsavev2_batch_normalization_33_moving_variance_read_readvariableop+savev2_conv2d_61_kernel_read_readvariableop)savev2_conv2d_61_bias_read_readvariableop+savev2_conv2d_62_kernel_read_readvariableop)savev2_conv2d_62_bias_read_readvariableop7savev2_batch_normalization_34_gamma_read_readvariableop6savev2_batch_normalization_34_beta_read_readvariableop=savev2_batch_normalization_34_moving_mean_read_readvariableopAsavev2_batch_normalization_34_moving_variance_read_readvariableop+savev2_conv2d_63_kernel_read_readvariableop)savev2_conv2d_63_bias_read_readvariableop7savev2_batch_normalization_35_gamma_read_readvariableop6savev2_batch_normalization_35_beta_read_readvariableop=savev2_batch_normalization_35_moving_mean_read_readvariableopAsavev2_batch_normalization_35_moving_variance_read_readvariableop*savev2_dense_22_kernel_read_readvariableop(savev2_dense_22_bias_read_readvariableop*savev2_dense_23_kernel_read_readvariableop(savev2_dense_23_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop2savev2_adam_conv2d_59_kernel_m_read_readvariableop0savev2_adam_conv2d_59_bias_m_read_readvariableop2savev2_adam_conv2d_60_kernel_m_read_readvariableop0savev2_adam_conv2d_60_bias_m_read_readvariableop>savev2_adam_batch_normalization_33_gamma_m_read_readvariableop=savev2_adam_batch_normalization_33_beta_m_read_readvariableop2savev2_adam_conv2d_61_kernel_m_read_readvariableop0savev2_adam_conv2d_61_bias_m_read_readvariableop2savev2_adam_conv2d_62_kernel_m_read_readvariableop0savev2_adam_conv2d_62_bias_m_read_readvariableop>savev2_adam_batch_normalization_34_gamma_m_read_readvariableop=savev2_adam_batch_normalization_34_beta_m_read_readvariableop2savev2_adam_conv2d_63_kernel_m_read_readvariableop0savev2_adam_conv2d_63_bias_m_read_readvariableop>savev2_adam_batch_normalization_35_gamma_m_read_readvariableop=savev2_adam_batch_normalization_35_beta_m_read_readvariableop1savev2_adam_dense_22_kernel_m_read_readvariableop/savev2_adam_dense_22_bias_m_read_readvariableop1savev2_adam_dense_23_kernel_m_read_readvariableop/savev2_adam_dense_23_bias_m_read_readvariableop2savev2_adam_conv2d_59_kernel_v_read_readvariableop0savev2_adam_conv2d_59_bias_v_read_readvariableop2savev2_adam_conv2d_60_kernel_v_read_readvariableop0savev2_adam_conv2d_60_bias_v_read_readvariableop>savev2_adam_batch_normalization_33_gamma_v_read_readvariableop=savev2_adam_batch_normalization_33_beta_v_read_readvariableop2savev2_adam_conv2d_61_kernel_v_read_readvariableop0savev2_adam_conv2d_61_bias_v_read_readvariableop2savev2_adam_conv2d_62_kernel_v_read_readvariableop0savev2_adam_conv2d_62_bias_v_read_readvariableop>savev2_adam_batch_normalization_34_gamma_v_read_readvariableop=savev2_adam_batch_normalization_34_beta_v_read_readvariableop2savev2_adam_conv2d_63_kernel_v_read_readvariableop0savev2_adam_conv2d_63_bias_v_read_readvariableop>savev2_adam_batch_normalization_35_gamma_v_read_readvariableop=savev2_adam_batch_normalization_35_beta_v_read_readvariableop1savev2_adam_dense_22_kernel_v_read_readvariableop/savev2_adam_dense_22_bias_v_read_readvariableop1savev2_adam_dense_23_kernel_v_read_readvariableop/savev2_adam_dense_23_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *Z
dtypesP
N2L	?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*?
_input_shapes?
?: :@:@:@@:@:@:@:@:@:@?:?:??:?:?:?:?:?:??:?:?:?:?:?:
??:?:	?:: : : : : : : : : :@:@:@@:@:@:@:@?:?:??:?:?:?:??:?:?:?:
??:?:	?::@:@:@@:@:@:@:@?:?:??:?:?:?:??:?:?:?:
??:?:	?:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
:@: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@: 

_output_shapes
:@:-	)
'
_output_shapes
:@?:!


_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:!

_output_shapes	
:?:&"
 
_output_shapes
:
??:!

_output_shapes	
:?:%!

_output_shapes
:	?: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
: :!

_output_shapes
: :"

_output_shapes
: :#

_output_shapes
: :,$(
&
_output_shapes
:@: %

_output_shapes
:@:,&(
&
_output_shapes
:@@: '

_output_shapes
:@: (

_output_shapes
:@: )

_output_shapes
:@:-*)
'
_output_shapes
:@?:!+

_output_shapes	
:?:.,*
(
_output_shapes
:??:!-

_output_shapes	
:?:!.

_output_shapes	
:?:!/

_output_shapes	
:?:.0*
(
_output_shapes
:??:!1

_output_shapes	
:?:!2

_output_shapes	
:?:!3

_output_shapes	
:?:&4"
 
_output_shapes
:
??:!5

_output_shapes	
:?:%6!

_output_shapes
:	?: 7

_output_shapes
::,8(
&
_output_shapes
:@: 9

_output_shapes
:@:,:(
&
_output_shapes
:@@: ;

_output_shapes
:@: <

_output_shapes
:@: =

_output_shapes
:@:->)
'
_output_shapes
:@?:!?

_output_shapes	
:?:.@*
(
_output_shapes
:??:!A

_output_shapes	
:?:!B

_output_shapes	
:?:!C

_output_shapes	
:?:.D*
(
_output_shapes
:??:!E

_output_shapes	
:?:!F

_output_shapes	
:?:!G

_output_shapes	
:?:&H"
 
_output_shapes
:
??:!I

_output_shapes	
:?:%J!

_output_shapes
:	?: K

_output_shapes
::L

_output_shapes
: 
?
i
M__inference_max_pooling2d_33_layer_call_and_return_conditional_losses_2445839

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?H
?
J__inference_sequential_13_layer_call_and_return_conditional_losses_2445158

inputs+
conv2d_59_2445091:@
conv2d_59_2445093:@+
conv2d_60_2445096:@@
conv2d_60_2445098:@,
batch_normalization_33_2445102:@,
batch_normalization_33_2445104:@,
batch_normalization_33_2445106:@,
batch_normalization_33_2445108:@,
conv2d_61_2445111:@? 
conv2d_61_2445113:	?-
conv2d_62_2445116:?? 
conv2d_62_2445118:	?-
batch_normalization_34_2445122:	?-
batch_normalization_34_2445124:	?-
batch_normalization_34_2445126:	?-
batch_normalization_34_2445128:	?-
conv2d_63_2445131:?? 
conv2d_63_2445133:	?-
batch_normalization_35_2445137:	?-
batch_normalization_35_2445139:	?-
batch_normalization_35_2445141:	?-
batch_normalization_35_2445143:	?$
dense_22_2445147:
??
dense_22_2445149:	?#
dense_23_2445152:	?
dense_23_2445154:
identity??.batch_normalization_33/StatefulPartitionedCall?.batch_normalization_34/StatefulPartitionedCall?.batch_normalization_35/StatefulPartitionedCall?!conv2d_59/StatefulPartitionedCall?!conv2d_60/StatefulPartitionedCall?!conv2d_61/StatefulPartitionedCall?!conv2d_62/StatefulPartitionedCall?!conv2d_63/StatefulPartitionedCall? dense_22/StatefulPartitionedCall? dense_23/StatefulPartitionedCall?
!conv2d_59/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_59_2445091conv2d_59_2445093*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????6@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_59_layer_call_and_return_conditional_losses_2444751?
!conv2d_60/StatefulPartitionedCallStatefulPartitionedCall*conv2d_59/StatefulPartitionedCall:output:0conv2d_60_2445096conv2d_60_2445098*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????4@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_60_layer_call_and_return_conditional_losses_2444768?
 max_pooling2d_33/PartitionedCallPartitionedCall*conv2d_60/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_max_pooling2d_33_layer_call_and_return_conditional_losses_2444514?
.batch_normalization_33/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_33/PartitionedCall:output:0batch_normalization_33_2445102batch_normalization_33_2445104batch_normalization_33_2445106batch_normalization_33_2445108*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_33_layer_call_and_return_conditional_losses_2444570?
!conv2d_61/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_33/StatefulPartitionedCall:output:0conv2d_61_2445111conv2d_61_2445113*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????
?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_61_layer_call_and_return_conditional_losses_2444795?
!conv2d_62/StatefulPartitionedCallStatefulPartitionedCall*conv2d_61/StatefulPartitionedCall:output:0conv2d_62_2445116conv2d_62_2445118*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_62_layer_call_and_return_conditional_losses_2444812?
 max_pooling2d_34/PartitionedCallPartitionedCall*conv2d_62/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_max_pooling2d_34_layer_call_and_return_conditional_losses_2444590?
.batch_normalization_34/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_34/PartitionedCall:output:0batch_normalization_34_2445122batch_normalization_34_2445124batch_normalization_34_2445126batch_normalization_34_2445128*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_34_layer_call_and_return_conditional_losses_2444646?
!conv2d_63/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_34/StatefulPartitionedCall:output:0conv2d_63_2445131conv2d_63_2445133*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????	?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_63_layer_call_and_return_conditional_losses_2444839?
 max_pooling2d_35/PartitionedCallPartitionedCall*conv2d_63/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_max_pooling2d_35_layer_call_and_return_conditional_losses_2444666?
.batch_normalization_35/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_35/PartitionedCall:output:0batch_normalization_35_2445137batch_normalization_35_2445139batch_normalization_35_2445141batch_normalization_35_2445143*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_35_layer_call_and_return_conditional_losses_2444722?
flatten_11/PartitionedCallPartitionedCall7batch_normalization_35/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_flatten_11_layer_call_and_return_conditional_losses_2444861?
 dense_22/StatefulPartitionedCallStatefulPartitionedCall#flatten_11/PartitionedCall:output:0dense_22_2445147dense_22_2445149*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_22_layer_call_and_return_conditional_losses_2444874?
 dense_23/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0dense_23_2445152dense_23_2445154*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_23_layer_call_and_return_conditional_losses_2444891x
IdentityIdentity)dense_23/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp/^batch_normalization_33/StatefulPartitionedCall/^batch_normalization_34/StatefulPartitionedCall/^batch_normalization_35/StatefulPartitionedCall"^conv2d_59/StatefulPartitionedCall"^conv2d_60/StatefulPartitionedCall"^conv2d_61/StatefulPartitionedCall"^conv2d_62/StatefulPartitionedCall"^conv2d_63/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:?????????8: : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_33/StatefulPartitionedCall.batch_normalization_33/StatefulPartitionedCall2`
.batch_normalization_34/StatefulPartitionedCall.batch_normalization_34/StatefulPartitionedCall2`
.batch_normalization_35/StatefulPartitionedCall.batch_normalization_35/StatefulPartitionedCall2F
!conv2d_59/StatefulPartitionedCall!conv2d_59/StatefulPartitionedCall2F
!conv2d_60/StatefulPartitionedCall!conv2d_60/StatefulPartitionedCall2F
!conv2d_61/StatefulPartitionedCall!conv2d_61/StatefulPartitionedCall2F
!conv2d_62/StatefulPartitionedCall!conv2d_62/StatefulPartitionedCall2F
!conv2d_63/StatefulPartitionedCall!conv2d_63/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall:W S
/
_output_shapes
:?????????8
 
_user_specified_nameinputs
?
N
2__inference_max_pooling2d_35_layer_call_fn_2446038

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_max_pooling2d_35_layer_call_and_return_conditional_losses_2444666?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

?
E__inference_dense_23_layer_call_and_return_conditional_losses_2446156

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_33_layer_call_fn_2445852

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_33_layer_call_and_return_conditional_losses_2444539?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_34_layer_call_and_return_conditional_losses_2444615

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
+__inference_conv2d_61_layer_call_fn_2445910

inputs"
unknown:@?
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????
?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_61_layer_call_and_return_conditional_losses_2444795x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????
?`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
??
?1
#__inference__traced_restore_2446639
file_prefix;
!assignvariableop_conv2d_59_kernel:@/
!assignvariableop_1_conv2d_59_bias:@=
#assignvariableop_2_conv2d_60_kernel:@@/
!assignvariableop_3_conv2d_60_bias:@=
/assignvariableop_4_batch_normalization_33_gamma:@<
.assignvariableop_5_batch_normalization_33_beta:@C
5assignvariableop_6_batch_normalization_33_moving_mean:@G
9assignvariableop_7_batch_normalization_33_moving_variance:@>
#assignvariableop_8_conv2d_61_kernel:@?0
!assignvariableop_9_conv2d_61_bias:	?@
$assignvariableop_10_conv2d_62_kernel:??1
"assignvariableop_11_conv2d_62_bias:	??
0assignvariableop_12_batch_normalization_34_gamma:	?>
/assignvariableop_13_batch_normalization_34_beta:	?E
6assignvariableop_14_batch_normalization_34_moving_mean:	?I
:assignvariableop_15_batch_normalization_34_moving_variance:	?@
$assignvariableop_16_conv2d_63_kernel:??1
"assignvariableop_17_conv2d_63_bias:	??
0assignvariableop_18_batch_normalization_35_gamma:	?>
/assignvariableop_19_batch_normalization_35_beta:	?E
6assignvariableop_20_batch_normalization_35_moving_mean:	?I
:assignvariableop_21_batch_normalization_35_moving_variance:	?7
#assignvariableop_22_dense_22_kernel:
??0
!assignvariableop_23_dense_22_bias:	?6
#assignvariableop_24_dense_23_kernel:	?/
!assignvariableop_25_dense_23_bias:'
assignvariableop_26_adam_iter:	 )
assignvariableop_27_adam_beta_1: )
assignvariableop_28_adam_beta_2: (
assignvariableop_29_adam_decay: 0
&assignvariableop_30_adam_learning_rate: %
assignvariableop_31_total_1: %
assignvariableop_32_count_1: #
assignvariableop_33_total: #
assignvariableop_34_count: E
+assignvariableop_35_adam_conv2d_59_kernel_m:@7
)assignvariableop_36_adam_conv2d_59_bias_m:@E
+assignvariableop_37_adam_conv2d_60_kernel_m:@@7
)assignvariableop_38_adam_conv2d_60_bias_m:@E
7assignvariableop_39_adam_batch_normalization_33_gamma_m:@D
6assignvariableop_40_adam_batch_normalization_33_beta_m:@F
+assignvariableop_41_adam_conv2d_61_kernel_m:@?8
)assignvariableop_42_adam_conv2d_61_bias_m:	?G
+assignvariableop_43_adam_conv2d_62_kernel_m:??8
)assignvariableop_44_adam_conv2d_62_bias_m:	?F
7assignvariableop_45_adam_batch_normalization_34_gamma_m:	?E
6assignvariableop_46_adam_batch_normalization_34_beta_m:	?G
+assignvariableop_47_adam_conv2d_63_kernel_m:??8
)assignvariableop_48_adam_conv2d_63_bias_m:	?F
7assignvariableop_49_adam_batch_normalization_35_gamma_m:	?E
6assignvariableop_50_adam_batch_normalization_35_beta_m:	?>
*assignvariableop_51_adam_dense_22_kernel_m:
??7
(assignvariableop_52_adam_dense_22_bias_m:	?=
*assignvariableop_53_adam_dense_23_kernel_m:	?6
(assignvariableop_54_adam_dense_23_bias_m:E
+assignvariableop_55_adam_conv2d_59_kernel_v:@7
)assignvariableop_56_adam_conv2d_59_bias_v:@E
+assignvariableop_57_adam_conv2d_60_kernel_v:@@7
)assignvariableop_58_adam_conv2d_60_bias_v:@E
7assignvariableop_59_adam_batch_normalization_33_gamma_v:@D
6assignvariableop_60_adam_batch_normalization_33_beta_v:@F
+assignvariableop_61_adam_conv2d_61_kernel_v:@?8
)assignvariableop_62_adam_conv2d_61_bias_v:	?G
+assignvariableop_63_adam_conv2d_62_kernel_v:??8
)assignvariableop_64_adam_conv2d_62_bias_v:	?F
7assignvariableop_65_adam_batch_normalization_34_gamma_v:	?E
6assignvariableop_66_adam_batch_normalization_34_beta_v:	?G
+assignvariableop_67_adam_conv2d_63_kernel_v:??8
)assignvariableop_68_adam_conv2d_63_bias_v:	?F
7assignvariableop_69_adam_batch_normalization_35_gamma_v:	?E
6assignvariableop_70_adam_batch_normalization_35_beta_v:	?>
*assignvariableop_71_adam_dense_22_kernel_v:
??7
(assignvariableop_72_adam_dense_22_bias_v:	?=
*assignvariableop_73_adam_dense_23_kernel_v:	?6
(assignvariableop_74_adam_dense_23_bias_v:
identity_76??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_55?AssignVariableOp_56?AssignVariableOp_57?AssignVariableOp_58?AssignVariableOp_59?AssignVariableOp_6?AssignVariableOp_60?AssignVariableOp_61?AssignVariableOp_62?AssignVariableOp_63?AssignVariableOp_64?AssignVariableOp_65?AssignVariableOp_66?AssignVariableOp_67?AssignVariableOp_68?AssignVariableOp_69?AssignVariableOp_7?AssignVariableOp_70?AssignVariableOp_71?AssignVariableOp_72?AssignVariableOp_73?AssignVariableOp_74?AssignVariableOp_8?AssignVariableOp_9?)
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:L*
dtype0*?)
value?)B?)LB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:L*
dtype0*?
value?B?LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*Z
dtypesP
N2L	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_59_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_59_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_60_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_60_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp/assignvariableop_4_batch_normalization_33_gammaIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp.assignvariableop_5_batch_normalization_33_betaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp5assignvariableop_6_batch_normalization_33_moving_meanIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp9assignvariableop_7_batch_normalization_33_moving_varianceIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp#assignvariableop_8_conv2d_61_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp!assignvariableop_9_conv2d_61_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp$assignvariableop_10_conv2d_62_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp"assignvariableop_11_conv2d_62_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp0assignvariableop_12_batch_normalization_34_gammaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp/assignvariableop_13_batch_normalization_34_betaIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp6assignvariableop_14_batch_normalization_34_moving_meanIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp:assignvariableop_15_batch_normalization_34_moving_varianceIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp$assignvariableop_16_conv2d_63_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp"assignvariableop_17_conv2d_63_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp0assignvariableop_18_batch_normalization_35_gammaIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp/assignvariableop_19_batch_normalization_35_betaIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp6assignvariableop_20_batch_normalization_35_moving_meanIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp:assignvariableop_21_batch_normalization_35_moving_varianceIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp#assignvariableop_22_dense_22_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp!assignvariableop_23_dense_22_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp#assignvariableop_24_dense_23_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp!assignvariableop_25_dense_23_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0	*
_output_shapes
:?
AssignVariableOp_26AssignVariableOpassignvariableop_26_adam_iterIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_27AssignVariableOpassignvariableop_27_adam_beta_1Identity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_28AssignVariableOpassignvariableop_28_adam_beta_2Identity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_29AssignVariableOpassignvariableop_29_adam_decayIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_30AssignVariableOp&assignvariableop_30_adam_learning_rateIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_31AssignVariableOpassignvariableop_31_total_1Identity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_32AssignVariableOpassignvariableop_32_count_1Identity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_33AssignVariableOpassignvariableop_33_totalIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_34AssignVariableOpassignvariableop_34_countIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_35AssignVariableOp+assignvariableop_35_adam_conv2d_59_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOp)assignvariableop_36_adam_conv2d_59_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_37AssignVariableOp+assignvariableop_37_adam_conv2d_60_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_38AssignVariableOp)assignvariableop_38_adam_conv2d_60_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_39AssignVariableOp7assignvariableop_39_adam_batch_normalization_33_gamma_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_40AssignVariableOp6assignvariableop_40_adam_batch_normalization_33_beta_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_41AssignVariableOp+assignvariableop_41_adam_conv2d_61_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_42AssignVariableOp)assignvariableop_42_adam_conv2d_61_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_43AssignVariableOp+assignvariableop_43_adam_conv2d_62_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_44AssignVariableOp)assignvariableop_44_adam_conv2d_62_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_45AssignVariableOp7assignvariableop_45_adam_batch_normalization_34_gamma_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_46AssignVariableOp6assignvariableop_46_adam_batch_normalization_34_beta_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_47AssignVariableOp+assignvariableop_47_adam_conv2d_63_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_48AssignVariableOp)assignvariableop_48_adam_conv2d_63_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_49AssignVariableOp7assignvariableop_49_adam_batch_normalization_35_gamma_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_50AssignVariableOp6assignvariableop_50_adam_batch_normalization_35_beta_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_51AssignVariableOp*assignvariableop_51_adam_dense_22_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_52AssignVariableOp(assignvariableop_52_adam_dense_22_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_53AssignVariableOp*assignvariableop_53_adam_dense_23_kernel_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_54AssignVariableOp(assignvariableop_54_adam_dense_23_bias_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_55AssignVariableOp+assignvariableop_55_adam_conv2d_59_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_56AssignVariableOp)assignvariableop_56_adam_conv2d_59_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_57AssignVariableOp+assignvariableop_57_adam_conv2d_60_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_58AssignVariableOp)assignvariableop_58_adam_conv2d_60_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_59AssignVariableOp7assignvariableop_59_adam_batch_normalization_33_gamma_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_60AssignVariableOp6assignvariableop_60_adam_batch_normalization_33_beta_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_61AssignVariableOp+assignvariableop_61_adam_conv2d_61_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_62AssignVariableOp)assignvariableop_62_adam_conv2d_61_bias_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_63AssignVariableOp+assignvariableop_63_adam_conv2d_62_kernel_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_64AssignVariableOp)assignvariableop_64_adam_conv2d_62_bias_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_65AssignVariableOp7assignvariableop_65_adam_batch_normalization_34_gamma_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_66AssignVariableOp6assignvariableop_66_adam_batch_normalization_34_beta_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_67AssignVariableOp+assignvariableop_67_adam_conv2d_63_kernel_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_68AssignVariableOp)assignvariableop_68_adam_conv2d_63_bias_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_69AssignVariableOp7assignvariableop_69_adam_batch_normalization_35_gamma_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_70AssignVariableOp6assignvariableop_70_adam_batch_normalization_35_beta_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_71AssignVariableOp*assignvariableop_71_adam_dense_22_kernel_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_72AssignVariableOp(assignvariableop_72_adam_dense_22_bias_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_73AssignVariableOp*assignvariableop_73_adam_dense_23_kernel_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_74AssignVariableOp(assignvariableop_74_adam_dense_23_bias_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 ?
Identity_75Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_76IdentityIdentity_75:output:0^NoOp_1*
T0*
_output_shapes
: ?
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_76Identity_76:output:0*?
_input_shapes?
?: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632*
AssignVariableOp_64AssignVariableOp_642*
AssignVariableOp_65AssignVariableOp_652*
AssignVariableOp_66AssignVariableOp_662*
AssignVariableOp_67AssignVariableOp_672*
AssignVariableOp_68AssignVariableOp_682*
AssignVariableOp_69AssignVariableOp_692(
AssignVariableOp_7AssignVariableOp_72*
AssignVariableOp_70AssignVariableOp_702*
AssignVariableOp_71AssignVariableOp_712*
AssignVariableOp_72AssignVariableOp_722*
AssignVariableOp_73AssignVariableOp_732*
AssignVariableOp_74AssignVariableOp_742(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
S__inference_batch_normalization_35_layer_call_and_return_conditional_losses_2446105

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_35_layer_call_and_return_conditional_losses_2446087

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
F__inference_conv2d_59_layer_call_and_return_conditional_losses_2445809

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????6@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????6@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????6@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????6@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????8: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????8
 
_user_specified_nameinputs
?
i
M__inference_max_pooling2d_34_layer_call_and_return_conditional_losses_2445951

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
F__inference_conv2d_60_layer_call_and_return_conditional_losses_2445829

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????4@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????4@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????4@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????4@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????6@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????6@
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_33_layer_call_fn_2445865

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_33_layer_call_and_return_conditional_losses_2444570?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_33_layer_call_and_return_conditional_losses_2445901

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
+__inference_conv2d_60_layer_call_fn_2445818

inputs!
unknown:@@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????4@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_60_layer_call_and_return_conditional_losses_2444768w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:?????????4@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????6@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????6@
 
_user_specified_nameinputs
?
?
*__inference_dense_22_layer_call_fn_2446125

inputs
unknown:
??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_22_layer_call_and_return_conditional_losses_2444874p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:??????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
F__inference_conv2d_61_layer_call_and_return_conditional_losses_2444795

inputs9
conv2d_readvariableop_resource:@?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????
?*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????
?Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????
?j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????
?w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
E__inference_dense_22_layer_call_and_return_conditional_losses_2444874

inputs2
matmul_readvariableop_resource:
??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:??????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_35_layer_call_fn_2446056

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_35_layer_call_and_return_conditional_losses_2444691?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
/__inference_sequential_13_layer_call_fn_2445589

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@
	unknown_6:@$
	unknown_7:@?
	unknown_8:	?%
	unknown_9:??

unknown_10:	?

unknown_11:	?

unknown_12:	?

unknown_13:	?

unknown_14:	?&

unknown_15:??

unknown_16:	?

unknown_17:	?

unknown_18:	?

unknown_19:	?

unknown_20:	?

unknown_21:
??

unknown_22:	?

unknown_23:	?

unknown_24:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_sequential_13_layer_call_and_return_conditional_losses_2445158o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:?????????8: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????8
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_34_layer_call_fn_2445964

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_34_layer_call_and_return_conditional_losses_2444615?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?

?
E__inference_dense_23_layer_call_and_return_conditional_losses_2444891

inputs1
matmul_readvariableop_resource:	?-
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	?*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
c
G__inference_flatten_11_layer_call_and_return_conditional_losses_2444861

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
F__inference_conv2d_61_layer_call_and_return_conditional_losses_2445921

inputs9
conv2d_readvariableop_resource:@?.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????
?*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????
?Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????
?j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????
?w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
F__inference_conv2d_60_layer_call_and_return_conditional_losses_2444768

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????4@*
paddingVALID*
strides
r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0}
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????4@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????4@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:?????????4@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????6@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????6@
 
_user_specified_nameinputs
?I
?
J__inference_sequential_13_layer_call_and_return_conditional_losses_2445340
conv2d_59_input+
conv2d_59_2445273:@
conv2d_59_2445275:@+
conv2d_60_2445278:@@
conv2d_60_2445280:@,
batch_normalization_33_2445284:@,
batch_normalization_33_2445286:@,
batch_normalization_33_2445288:@,
batch_normalization_33_2445290:@,
conv2d_61_2445293:@? 
conv2d_61_2445295:	?-
conv2d_62_2445298:?? 
conv2d_62_2445300:	?-
batch_normalization_34_2445304:	?-
batch_normalization_34_2445306:	?-
batch_normalization_34_2445308:	?-
batch_normalization_34_2445310:	?-
conv2d_63_2445313:?? 
conv2d_63_2445315:	?-
batch_normalization_35_2445319:	?-
batch_normalization_35_2445321:	?-
batch_normalization_35_2445323:	?-
batch_normalization_35_2445325:	?$
dense_22_2445329:
??
dense_22_2445331:	?#
dense_23_2445334:	?
dense_23_2445336:
identity??.batch_normalization_33/StatefulPartitionedCall?.batch_normalization_34/StatefulPartitionedCall?.batch_normalization_35/StatefulPartitionedCall?!conv2d_59/StatefulPartitionedCall?!conv2d_60/StatefulPartitionedCall?!conv2d_61/StatefulPartitionedCall?!conv2d_62/StatefulPartitionedCall?!conv2d_63/StatefulPartitionedCall? dense_22/StatefulPartitionedCall? dense_23/StatefulPartitionedCall?
!conv2d_59/StatefulPartitionedCallStatefulPartitionedCallconv2d_59_inputconv2d_59_2445273conv2d_59_2445275*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????6@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_59_layer_call_and_return_conditional_losses_2444751?
!conv2d_60/StatefulPartitionedCallStatefulPartitionedCall*conv2d_59/StatefulPartitionedCall:output:0conv2d_60_2445278conv2d_60_2445280*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????4@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_60_layer_call_and_return_conditional_losses_2444768?
 max_pooling2d_33/PartitionedCallPartitionedCall*conv2d_60/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_max_pooling2d_33_layer_call_and_return_conditional_losses_2444514?
.batch_normalization_33/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_33/PartitionedCall:output:0batch_normalization_33_2445284batch_normalization_33_2445286batch_normalization_33_2445288batch_normalization_33_2445290*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_33_layer_call_and_return_conditional_losses_2444539?
!conv2d_61/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_33/StatefulPartitionedCall:output:0conv2d_61_2445293conv2d_61_2445295*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????
?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_61_layer_call_and_return_conditional_losses_2444795?
!conv2d_62/StatefulPartitionedCallStatefulPartitionedCall*conv2d_61/StatefulPartitionedCall:output:0conv2d_62_2445298conv2d_62_2445300*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_62_layer_call_and_return_conditional_losses_2444812?
 max_pooling2d_34/PartitionedCallPartitionedCall*conv2d_62/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_max_pooling2d_34_layer_call_and_return_conditional_losses_2444590?
.batch_normalization_34/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_34/PartitionedCall:output:0batch_normalization_34_2445304batch_normalization_34_2445306batch_normalization_34_2445308batch_normalization_34_2445310*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_34_layer_call_and_return_conditional_losses_2444615?
!conv2d_63/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_34/StatefulPartitionedCall:output:0conv2d_63_2445313conv2d_63_2445315*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????	?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_63_layer_call_and_return_conditional_losses_2444839?
 max_pooling2d_35/PartitionedCallPartitionedCall*conv2d_63/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_max_pooling2d_35_layer_call_and_return_conditional_losses_2444666?
.batch_normalization_35/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_35/PartitionedCall:output:0batch_normalization_35_2445319batch_normalization_35_2445321batch_normalization_35_2445323batch_normalization_35_2445325*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_35_layer_call_and_return_conditional_losses_2444691?
flatten_11/PartitionedCallPartitionedCall7batch_normalization_35/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_flatten_11_layer_call_and_return_conditional_losses_2444861?
 dense_22/StatefulPartitionedCallStatefulPartitionedCall#flatten_11/PartitionedCall:output:0dense_22_2445329dense_22_2445331*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_22_layer_call_and_return_conditional_losses_2444874?
 dense_23/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0dense_23_2445334dense_23_2445336*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_23_layer_call_and_return_conditional_losses_2444891x
IdentityIdentity)dense_23/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp/^batch_normalization_33/StatefulPartitionedCall/^batch_normalization_34/StatefulPartitionedCall/^batch_normalization_35/StatefulPartitionedCall"^conv2d_59/StatefulPartitionedCall"^conv2d_60/StatefulPartitionedCall"^conv2d_61/StatefulPartitionedCall"^conv2d_62/StatefulPartitionedCall"^conv2d_63/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:?????????8: : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_33/StatefulPartitionedCall.batch_normalization_33/StatefulPartitionedCall2`
.batch_normalization_34/StatefulPartitionedCall.batch_normalization_34/StatefulPartitionedCall2`
.batch_normalization_35/StatefulPartitionedCall.batch_normalization_35/StatefulPartitionedCall2F
!conv2d_59/StatefulPartitionedCall!conv2d_59/StatefulPartitionedCall2F
!conv2d_60/StatefulPartitionedCall!conv2d_60/StatefulPartitionedCall2F
!conv2d_61/StatefulPartitionedCall!conv2d_61/StatefulPartitionedCall2F
!conv2d_62/StatefulPartitionedCall!conv2d_62/StatefulPartitionedCall2F
!conv2d_63/StatefulPartitionedCall!conv2d_63/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall:` \
/
_output_shapes
:?????????8
)
_user_specified_nameconv2d_59_input
?
?
/__inference_sequential_13_layer_call_fn_2444953
conv2d_59_input!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@
	unknown_6:@$
	unknown_7:@?
	unknown_8:	?%
	unknown_9:??

unknown_10:	?

unknown_11:	?

unknown_12:	?

unknown_13:	?

unknown_14:	?&

unknown_15:??

unknown_16:	?

unknown_17:	?

unknown_18:	?

unknown_19:	?

unknown_20:	?

unknown_21:
??

unknown_22:	?

unknown_23:	?

unknown_24:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_59_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *S
fNRL
J__inference_sequential_13_layer_call_and_return_conditional_losses_2444898o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:?????????8: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:?????????8
)
_user_specified_nameconv2d_59_input
?
?
S__inference_batch_normalization_33_layer_call_and_return_conditional_losses_2444570

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+???????????????????????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+???????????????????????????@?
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+???????????????????????????@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
H
,__inference_flatten_11_layer_call_fn_2446110

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_flatten_11_layer_call_and_return_conditional_losses_2444861a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
F__inference_conv2d_62_layer_call_and_return_conditional_losses_2445941

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:??????????j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:??????????w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :?????????
?: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:?????????
?
 
_user_specified_nameinputs
?
?
+__inference_conv2d_63_layer_call_fn_2446022

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????	?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_63_layer_call_and_return_conditional_losses_2444839x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:?????????	?`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
%__inference_signature_wrapper_2445475
conv2d_59_input!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@
	unknown_6:@$
	unknown_7:@?
	unknown_8:	?%
	unknown_9:??

unknown_10:	?

unknown_11:	?

unknown_12:	?

unknown_13:	?

unknown_14:	?&

unknown_15:??

unknown_16:	?

unknown_17:	?

unknown_18:	?

unknown_19:	?

unknown_20:	?

unknown_21:
??

unknown_22:	?

unknown_23:	?

unknown_24:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallconv2d_59_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *+
f&R$
"__inference__wrapped_model_2444505o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:?????????8: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:?????????8
)
_user_specified_nameconv2d_59_input
?
?
*__inference_dense_23_layer_call_fn_2446145

inputs
unknown:	?
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_23_layer_call_and_return_conditional_losses_2444891o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?I
?
J__inference_sequential_13_layer_call_and_return_conditional_losses_2445410
conv2d_59_input+
conv2d_59_2445343:@
conv2d_59_2445345:@+
conv2d_60_2445348:@@
conv2d_60_2445350:@,
batch_normalization_33_2445354:@,
batch_normalization_33_2445356:@,
batch_normalization_33_2445358:@,
batch_normalization_33_2445360:@,
conv2d_61_2445363:@? 
conv2d_61_2445365:	?-
conv2d_62_2445368:?? 
conv2d_62_2445370:	?-
batch_normalization_34_2445374:	?-
batch_normalization_34_2445376:	?-
batch_normalization_34_2445378:	?-
batch_normalization_34_2445380:	?-
conv2d_63_2445383:?? 
conv2d_63_2445385:	?-
batch_normalization_35_2445389:	?-
batch_normalization_35_2445391:	?-
batch_normalization_35_2445393:	?-
batch_normalization_35_2445395:	?$
dense_22_2445399:
??
dense_22_2445401:	?#
dense_23_2445404:	?
dense_23_2445406:
identity??.batch_normalization_33/StatefulPartitionedCall?.batch_normalization_34/StatefulPartitionedCall?.batch_normalization_35/StatefulPartitionedCall?!conv2d_59/StatefulPartitionedCall?!conv2d_60/StatefulPartitionedCall?!conv2d_61/StatefulPartitionedCall?!conv2d_62/StatefulPartitionedCall?!conv2d_63/StatefulPartitionedCall? dense_22/StatefulPartitionedCall? dense_23/StatefulPartitionedCall?
!conv2d_59/StatefulPartitionedCallStatefulPartitionedCallconv2d_59_inputconv2d_59_2445343conv2d_59_2445345*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????6@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_59_layer_call_and_return_conditional_losses_2444751?
!conv2d_60/StatefulPartitionedCallStatefulPartitionedCall*conv2d_59/StatefulPartitionedCall:output:0conv2d_60_2445348conv2d_60_2445350*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????4@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_60_layer_call_and_return_conditional_losses_2444768?
 max_pooling2d_33/PartitionedCallPartitionedCall*conv2d_60/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_max_pooling2d_33_layer_call_and_return_conditional_losses_2444514?
.batch_normalization_33/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_33/PartitionedCall:output:0batch_normalization_33_2445354batch_normalization_33_2445356batch_normalization_33_2445358batch_normalization_33_2445360*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_33_layer_call_and_return_conditional_losses_2444570?
!conv2d_61/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_33/StatefulPartitionedCall:output:0conv2d_61_2445363conv2d_61_2445365*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????
?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_61_layer_call_and_return_conditional_losses_2444795?
!conv2d_62/StatefulPartitionedCallStatefulPartitionedCall*conv2d_61/StatefulPartitionedCall:output:0conv2d_62_2445368conv2d_62_2445370*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_62_layer_call_and_return_conditional_losses_2444812?
 max_pooling2d_34/PartitionedCallPartitionedCall*conv2d_62/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_max_pooling2d_34_layer_call_and_return_conditional_losses_2444590?
.batch_normalization_34/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_34/PartitionedCall:output:0batch_normalization_34_2445374batch_normalization_34_2445376batch_normalization_34_2445378batch_normalization_34_2445380*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_34_layer_call_and_return_conditional_losses_2444646?
!conv2d_63/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_34/StatefulPartitionedCall:output:0conv2d_63_2445383conv2d_63_2445385*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????	?*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_63_layer_call_and_return_conditional_losses_2444839?
 max_pooling2d_35/PartitionedCallPartitionedCall*conv2d_63/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_max_pooling2d_35_layer_call_and_return_conditional_losses_2444666?
.batch_normalization_35/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_35/PartitionedCall:output:0batch_normalization_35_2445389batch_normalization_35_2445391batch_normalization_35_2445393batch_normalization_35_2445395*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_35_layer_call_and_return_conditional_losses_2444722?
flatten_11/PartitionedCallPartitionedCall7batch_normalization_35/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_flatten_11_layer_call_and_return_conditional_losses_2444861?
 dense_22/StatefulPartitionedCallStatefulPartitionedCall#flatten_11/PartitionedCall:output:0dense_22_2445399dense_22_2445401*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_22_layer_call_and_return_conditional_losses_2444874?
 dense_23/StatefulPartitionedCallStatefulPartitionedCall)dense_22/StatefulPartitionedCall:output:0dense_23_2445404dense_23_2445406*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_dense_23_layer_call_and_return_conditional_losses_2444891x
IdentityIdentity)dense_23/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp/^batch_normalization_33/StatefulPartitionedCall/^batch_normalization_34/StatefulPartitionedCall/^batch_normalization_35/StatefulPartitionedCall"^conv2d_59/StatefulPartitionedCall"^conv2d_60/StatefulPartitionedCall"^conv2d_61/StatefulPartitionedCall"^conv2d_62/StatefulPartitionedCall"^conv2d_63/StatefulPartitionedCall!^dense_22/StatefulPartitionedCall!^dense_23/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:?????????8: : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_33/StatefulPartitionedCall.batch_normalization_33/StatefulPartitionedCall2`
.batch_normalization_34/StatefulPartitionedCall.batch_normalization_34/StatefulPartitionedCall2`
.batch_normalization_35/StatefulPartitionedCall.batch_normalization_35/StatefulPartitionedCall2F
!conv2d_59/StatefulPartitionedCall!conv2d_59/StatefulPartitionedCall2F
!conv2d_60/StatefulPartitionedCall!conv2d_60/StatefulPartitionedCall2F
!conv2d_61/StatefulPartitionedCall!conv2d_61/StatefulPartitionedCall2F
!conv2d_62/StatefulPartitionedCall!conv2d_62/StatefulPartitionedCall2F
!conv2d_63/StatefulPartitionedCall!conv2d_63/StatefulPartitionedCall2D
 dense_22/StatefulPartitionedCall dense_22/StatefulPartitionedCall2D
 dense_23/StatefulPartitionedCall dense_23/StatefulPartitionedCall:` \
/
_output_shapes
:?????????8
)
_user_specified_nameconv2d_59_input
?
?
S__inference_batch_normalization_34_layer_call_and_return_conditional_losses_2444646

inputs&
readvariableop_resource:	?(
readvariableop_1_resource:	?7
(fusedbatchnormv3_readvariableop_resource:	?9
*fusedbatchnormv3_readvariableop_1_resource:	?
identity??AssignNewValue?AssignNewValue_1?FusedBatchNormV3/ReadVariableOp?!FusedBatchNormV3/ReadVariableOp_1?ReadVariableOp?ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:?*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,????????????????????????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,?????????????????????????????
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
i
M__inference_max_pooling2d_35_layer_call_and_return_conditional_losses_2444666

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
N
2__inference_max_pooling2d_34_layer_call_fn_2445946

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_max_pooling2d_34_layer_call_and_return_conditional_losses_2444590?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
F__inference_conv2d_63_layer_call_and_return_conditional_losses_2446033

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	?*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	?Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:?????????	?j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:?????????	?w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
c
G__inference_flatten_11_layer_call_and_return_conditional_losses_2446116

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
??
?
J__inference_sequential_13_layer_call_and_return_conditional_losses_2445789

inputsB
(conv2d_59_conv2d_readvariableop_resource:@7
)conv2d_59_biasadd_readvariableop_resource:@B
(conv2d_60_conv2d_readvariableop_resource:@@7
)conv2d_60_biasadd_readvariableop_resource:@<
.batch_normalization_33_readvariableop_resource:@>
0batch_normalization_33_readvariableop_1_resource:@M
?batch_normalization_33_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_33_fusedbatchnormv3_readvariableop_1_resource:@C
(conv2d_61_conv2d_readvariableop_resource:@?8
)conv2d_61_biasadd_readvariableop_resource:	?D
(conv2d_62_conv2d_readvariableop_resource:??8
)conv2d_62_biasadd_readvariableop_resource:	?=
.batch_normalization_34_readvariableop_resource:	??
0batch_normalization_34_readvariableop_1_resource:	?N
?batch_normalization_34_fusedbatchnormv3_readvariableop_resource:	?P
Abatch_normalization_34_fusedbatchnormv3_readvariableop_1_resource:	?D
(conv2d_63_conv2d_readvariableop_resource:??8
)conv2d_63_biasadd_readvariableop_resource:	?=
.batch_normalization_35_readvariableop_resource:	??
0batch_normalization_35_readvariableop_1_resource:	?N
?batch_normalization_35_fusedbatchnormv3_readvariableop_resource:	?P
Abatch_normalization_35_fusedbatchnormv3_readvariableop_1_resource:	?;
'dense_22_matmul_readvariableop_resource:
??7
(dense_22_biasadd_readvariableop_resource:	?:
'dense_23_matmul_readvariableop_resource:	?6
(dense_23_biasadd_readvariableop_resource:
identity??%batch_normalization_33/AssignNewValue?'batch_normalization_33/AssignNewValue_1?6batch_normalization_33/FusedBatchNormV3/ReadVariableOp?8batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_33/ReadVariableOp?'batch_normalization_33/ReadVariableOp_1?%batch_normalization_34/AssignNewValue?'batch_normalization_34/AssignNewValue_1?6batch_normalization_34/FusedBatchNormV3/ReadVariableOp?8batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_34/ReadVariableOp?'batch_normalization_34/ReadVariableOp_1?%batch_normalization_35/AssignNewValue?'batch_normalization_35/AssignNewValue_1?6batch_normalization_35/FusedBatchNormV3/ReadVariableOp?8batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_35/ReadVariableOp?'batch_normalization_35/ReadVariableOp_1? conv2d_59/BiasAdd/ReadVariableOp?conv2d_59/Conv2D/ReadVariableOp? conv2d_60/BiasAdd/ReadVariableOp?conv2d_60/Conv2D/ReadVariableOp? conv2d_61/BiasAdd/ReadVariableOp?conv2d_61/Conv2D/ReadVariableOp? conv2d_62/BiasAdd/ReadVariableOp?conv2d_62/Conv2D/ReadVariableOp? conv2d_63/BiasAdd/ReadVariableOp?conv2d_63/Conv2D/ReadVariableOp?dense_22/BiasAdd/ReadVariableOp?dense_22/MatMul/ReadVariableOp?dense_23/BiasAdd/ReadVariableOp?dense_23/MatMul/ReadVariableOp?
conv2d_59/Conv2D/ReadVariableOpReadVariableOp(conv2d_59_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
conv2d_59/Conv2DConv2Dinputs'conv2d_59/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????6@*
paddingVALID*
strides
?
 conv2d_59/BiasAdd/ReadVariableOpReadVariableOp)conv2d_59_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_59/BiasAddBiasAddconv2d_59/Conv2D:output:0(conv2d_59/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????6@l
conv2d_59/ReluReluconv2d_59/BiasAdd:output:0*
T0*/
_output_shapes
:?????????6@?
conv2d_60/Conv2D/ReadVariableOpReadVariableOp(conv2d_60_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
conv2d_60/Conv2DConv2Dconv2d_59/Relu:activations:0'conv2d_60/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????4@*
paddingVALID*
strides
?
 conv2d_60/BiasAdd/ReadVariableOpReadVariableOp)conv2d_60_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_60/BiasAddBiasAddconv2d_60/Conv2D:output:0(conv2d_60/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????4@l
conv2d_60/ReluReluconv2d_60/BiasAdd:output:0*
T0*/
_output_shapes
:?????????4@?
max_pooling2d_33/MaxPoolMaxPoolconv2d_60/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
?
%batch_normalization_33/ReadVariableOpReadVariableOp.batch_normalization_33_readvariableop_resource*
_output_shapes
:@*
dtype0?
'batch_normalization_33/ReadVariableOp_1ReadVariableOp0batch_normalization_33_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
6batch_normalization_33/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_33_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
8batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_33_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
'batch_normalization_33/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_33/MaxPool:output:0-batch_normalization_33/ReadVariableOp:value:0/batch_normalization_33/ReadVariableOp_1:value:0>batch_normalization_33/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<?
%batch_normalization_33/AssignNewValueAssignVariableOp?batch_normalization_33_fusedbatchnormv3_readvariableop_resource4batch_normalization_33/FusedBatchNormV3:batch_mean:07^batch_normalization_33/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
'batch_normalization_33/AssignNewValue_1AssignVariableOpAbatch_normalization_33_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_33/FusedBatchNormV3:batch_variance:09^batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(?
conv2d_61/Conv2D/ReadVariableOpReadVariableOp(conv2d_61_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
conv2d_61/Conv2DConv2D+batch_normalization_33/FusedBatchNormV3:y:0'conv2d_61/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????
?*
paddingVALID*
strides
?
 conv2d_61/BiasAdd/ReadVariableOpReadVariableOp)conv2d_61_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
conv2d_61/BiasAddBiasAddconv2d_61/Conv2D:output:0(conv2d_61/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????
?m
conv2d_61/ReluReluconv2d_61/BiasAdd:output:0*
T0*0
_output_shapes
:?????????
??
conv2d_62/Conv2D/ReadVariableOpReadVariableOp(conv2d_62_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
conv2d_62/Conv2DConv2Dconv2d_61/Relu:activations:0'conv2d_62/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
?
 conv2d_62/BiasAdd/ReadVariableOpReadVariableOp)conv2d_62_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
conv2d_62/BiasAddBiasAddconv2d_62/Conv2D:output:0(conv2d_62/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????m
conv2d_62/ReluReluconv2d_62/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
max_pooling2d_34/MaxPoolMaxPoolconv2d_62/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
?
%batch_normalization_34/ReadVariableOpReadVariableOp.batch_normalization_34_readvariableop_resource*
_output_shapes	
:?*
dtype0?
'batch_normalization_34/ReadVariableOp_1ReadVariableOp0batch_normalization_34_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
6batch_normalization_34/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_34_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
8batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_34_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
'batch_normalization_34/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_34/MaxPool:output:0-batch_normalization_34/ReadVariableOp:value:0/batch_normalization_34/ReadVariableOp_1:value:0>batch_normalization_34/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
%batch_normalization_34/AssignNewValueAssignVariableOp?batch_normalization_34_fusedbatchnormv3_readvariableop_resource4batch_normalization_34/FusedBatchNormV3:batch_mean:07^batch_normalization_34/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
'batch_normalization_34/AssignNewValue_1AssignVariableOpAbatch_normalization_34_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_34/FusedBatchNormV3:batch_variance:09^batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(?
conv2d_63/Conv2D/ReadVariableOpReadVariableOp(conv2d_63_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
conv2d_63/Conv2DConv2D+batch_normalization_34/FusedBatchNormV3:y:0'conv2d_63/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	?*
paddingVALID*
strides
?
 conv2d_63/BiasAdd/ReadVariableOpReadVariableOp)conv2d_63_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
conv2d_63/BiasAddBiasAddconv2d_63/Conv2D:output:0(conv2d_63/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	?m
conv2d_63/ReluReluconv2d_63/BiasAdd:output:0*
T0*0
_output_shapes
:?????????	??
max_pooling2d_35/MaxPoolMaxPoolconv2d_63/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
?
%batch_normalization_35/ReadVariableOpReadVariableOp.batch_normalization_35_readvariableop_resource*
_output_shapes	
:?*
dtype0?
'batch_normalization_35/ReadVariableOp_1ReadVariableOp0batch_normalization_35_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
6batch_normalization_35/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_35_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
8batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_35_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
'batch_normalization_35/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_35/MaxPool:output:0-batch_normalization_35/ReadVariableOp:value:0/batch_normalization_35/ReadVariableOp_1:value:0>batch_normalization_35/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
%batch_normalization_35/AssignNewValueAssignVariableOp?batch_normalization_35_fusedbatchnormv3_readvariableop_resource4batch_normalization_35/FusedBatchNormV3:batch_mean:07^batch_normalization_35/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
'batch_normalization_35/AssignNewValue_1AssignVariableOpAbatch_normalization_35_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_35/FusedBatchNormV3:batch_variance:09^batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(a
flatten_11/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ?
flatten_11/ReshapeReshape+batch_normalization_35/FusedBatchNormV3:y:0flatten_11/Const:output:0*
T0*(
_output_shapes
:???????????
dense_22/MatMul/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense_22/MatMulMatMulflatten_11/Reshape:output:0&dense_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_22/BiasAdd/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_22/BiasAddBiasAdddense_22/MatMul:product:0'dense_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????c
dense_22/ReluReludense_22/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
dense_23/MatMulMatMuldense_22/Relu:activations:0&dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????h
dense_23/SoftmaxSoftmaxdense_23/BiasAdd:output:0*
T0*'
_output_shapes
:?????????i
IdentityIdentitydense_23/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????

NoOpNoOp&^batch_normalization_33/AssignNewValue(^batch_normalization_33/AssignNewValue_17^batch_normalization_33/FusedBatchNormV3/ReadVariableOp9^batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_33/ReadVariableOp(^batch_normalization_33/ReadVariableOp_1&^batch_normalization_34/AssignNewValue(^batch_normalization_34/AssignNewValue_17^batch_normalization_34/FusedBatchNormV3/ReadVariableOp9^batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_34/ReadVariableOp(^batch_normalization_34/ReadVariableOp_1&^batch_normalization_35/AssignNewValue(^batch_normalization_35/AssignNewValue_17^batch_normalization_35/FusedBatchNormV3/ReadVariableOp9^batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_35/ReadVariableOp(^batch_normalization_35/ReadVariableOp_1!^conv2d_59/BiasAdd/ReadVariableOp ^conv2d_59/Conv2D/ReadVariableOp!^conv2d_60/BiasAdd/ReadVariableOp ^conv2d_60/Conv2D/ReadVariableOp!^conv2d_61/BiasAdd/ReadVariableOp ^conv2d_61/Conv2D/ReadVariableOp!^conv2d_62/BiasAdd/ReadVariableOp ^conv2d_62/Conv2D/ReadVariableOp!^conv2d_63/BiasAdd/ReadVariableOp ^conv2d_63/Conv2D/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:?????????8: : : : : : : : : : : : : : : : : : : : : : : : : : 2N
%batch_normalization_33/AssignNewValue%batch_normalization_33/AssignNewValue2R
'batch_normalization_33/AssignNewValue_1'batch_normalization_33/AssignNewValue_12p
6batch_normalization_33/FusedBatchNormV3/ReadVariableOp6batch_normalization_33/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_33/FusedBatchNormV3/ReadVariableOp_18batch_normalization_33/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_33/ReadVariableOp%batch_normalization_33/ReadVariableOp2R
'batch_normalization_33/ReadVariableOp_1'batch_normalization_33/ReadVariableOp_12N
%batch_normalization_34/AssignNewValue%batch_normalization_34/AssignNewValue2R
'batch_normalization_34/AssignNewValue_1'batch_normalization_34/AssignNewValue_12p
6batch_normalization_34/FusedBatchNormV3/ReadVariableOp6batch_normalization_34/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_34/FusedBatchNormV3/ReadVariableOp_18batch_normalization_34/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_34/ReadVariableOp%batch_normalization_34/ReadVariableOp2R
'batch_normalization_34/ReadVariableOp_1'batch_normalization_34/ReadVariableOp_12N
%batch_normalization_35/AssignNewValue%batch_normalization_35/AssignNewValue2R
'batch_normalization_35/AssignNewValue_1'batch_normalization_35/AssignNewValue_12p
6batch_normalization_35/FusedBatchNormV3/ReadVariableOp6batch_normalization_35/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_35/FusedBatchNormV3/ReadVariableOp_18batch_normalization_35/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_35/ReadVariableOp%batch_normalization_35/ReadVariableOp2R
'batch_normalization_35/ReadVariableOp_1'batch_normalization_35/ReadVariableOp_12D
 conv2d_59/BiasAdd/ReadVariableOp conv2d_59/BiasAdd/ReadVariableOp2B
conv2d_59/Conv2D/ReadVariableOpconv2d_59/Conv2D/ReadVariableOp2D
 conv2d_60/BiasAdd/ReadVariableOp conv2d_60/BiasAdd/ReadVariableOp2B
conv2d_60/Conv2D/ReadVariableOpconv2d_60/Conv2D/ReadVariableOp2D
 conv2d_61/BiasAdd/ReadVariableOp conv2d_61/BiasAdd/ReadVariableOp2B
conv2d_61/Conv2D/ReadVariableOpconv2d_61/Conv2D/ReadVariableOp2D
 conv2d_62/BiasAdd/ReadVariableOp conv2d_62/BiasAdd/ReadVariableOp2B
conv2d_62/Conv2D/ReadVariableOpconv2d_62/Conv2D/ReadVariableOp2D
 conv2d_63/BiasAdd/ReadVariableOp conv2d_63/BiasAdd/ReadVariableOp2B
conv2d_63/Conv2D/ReadVariableOpconv2d_63/Conv2D/ReadVariableOp2B
dense_22/BiasAdd/ReadVariableOpdense_22/BiasAdd/ReadVariableOp2@
dense_22/MatMul/ReadVariableOpdense_22/MatMul/ReadVariableOp2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????8
 
_user_specified_nameinputs
?	
?
8__inference_batch_normalization_34_layer_call_fn_2445977

inputs
unknown:	?
	unknown_0:	?
	unknown_1:	?
	unknown_2:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *\
fWRU
S__inference_batch_normalization_34_layer_call_and_return_conditional_losses_2444646?
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,????????????????????????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,????????????????????????????: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?{
?
J__inference_sequential_13_layer_call_and_return_conditional_losses_2445689

inputsB
(conv2d_59_conv2d_readvariableop_resource:@7
)conv2d_59_biasadd_readvariableop_resource:@B
(conv2d_60_conv2d_readvariableop_resource:@@7
)conv2d_60_biasadd_readvariableop_resource:@<
.batch_normalization_33_readvariableop_resource:@>
0batch_normalization_33_readvariableop_1_resource:@M
?batch_normalization_33_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_33_fusedbatchnormv3_readvariableop_1_resource:@C
(conv2d_61_conv2d_readvariableop_resource:@?8
)conv2d_61_biasadd_readvariableop_resource:	?D
(conv2d_62_conv2d_readvariableop_resource:??8
)conv2d_62_biasadd_readvariableop_resource:	?=
.batch_normalization_34_readvariableop_resource:	??
0batch_normalization_34_readvariableop_1_resource:	?N
?batch_normalization_34_fusedbatchnormv3_readvariableop_resource:	?P
Abatch_normalization_34_fusedbatchnormv3_readvariableop_1_resource:	?D
(conv2d_63_conv2d_readvariableop_resource:??8
)conv2d_63_biasadd_readvariableop_resource:	?=
.batch_normalization_35_readvariableop_resource:	??
0batch_normalization_35_readvariableop_1_resource:	?N
?batch_normalization_35_fusedbatchnormv3_readvariableop_resource:	?P
Abatch_normalization_35_fusedbatchnormv3_readvariableop_1_resource:	?;
'dense_22_matmul_readvariableop_resource:
??7
(dense_22_biasadd_readvariableop_resource:	?:
'dense_23_matmul_readvariableop_resource:	?6
(dense_23_biasadd_readvariableop_resource:
identity??6batch_normalization_33/FusedBatchNormV3/ReadVariableOp?8batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_33/ReadVariableOp?'batch_normalization_33/ReadVariableOp_1?6batch_normalization_34/FusedBatchNormV3/ReadVariableOp?8batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_34/ReadVariableOp?'batch_normalization_34/ReadVariableOp_1?6batch_normalization_35/FusedBatchNormV3/ReadVariableOp?8batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_35/ReadVariableOp?'batch_normalization_35/ReadVariableOp_1? conv2d_59/BiasAdd/ReadVariableOp?conv2d_59/Conv2D/ReadVariableOp? conv2d_60/BiasAdd/ReadVariableOp?conv2d_60/Conv2D/ReadVariableOp? conv2d_61/BiasAdd/ReadVariableOp?conv2d_61/Conv2D/ReadVariableOp? conv2d_62/BiasAdd/ReadVariableOp?conv2d_62/Conv2D/ReadVariableOp? conv2d_63/BiasAdd/ReadVariableOp?conv2d_63/Conv2D/ReadVariableOp?dense_22/BiasAdd/ReadVariableOp?dense_22/MatMul/ReadVariableOp?dense_23/BiasAdd/ReadVariableOp?dense_23/MatMul/ReadVariableOp?
conv2d_59/Conv2D/ReadVariableOpReadVariableOp(conv2d_59_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
conv2d_59/Conv2DConv2Dinputs'conv2d_59/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????6@*
paddingVALID*
strides
?
 conv2d_59/BiasAdd/ReadVariableOpReadVariableOp)conv2d_59_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_59/BiasAddBiasAddconv2d_59/Conv2D:output:0(conv2d_59/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????6@l
conv2d_59/ReluReluconv2d_59/BiasAdd:output:0*
T0*/
_output_shapes
:?????????6@?
conv2d_60/Conv2D/ReadVariableOpReadVariableOp(conv2d_60_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
conv2d_60/Conv2DConv2Dconv2d_59/Relu:activations:0'conv2d_60/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????4@*
paddingVALID*
strides
?
 conv2d_60/BiasAdd/ReadVariableOpReadVariableOp)conv2d_60_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_60/BiasAddBiasAddconv2d_60/Conv2D:output:0(conv2d_60/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????4@l
conv2d_60/ReluReluconv2d_60/BiasAdd:output:0*
T0*/
_output_shapes
:?????????4@?
max_pooling2d_33/MaxPoolMaxPoolconv2d_60/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
?
%batch_normalization_33/ReadVariableOpReadVariableOp.batch_normalization_33_readvariableop_resource*
_output_shapes
:@*
dtype0?
'batch_normalization_33/ReadVariableOp_1ReadVariableOp0batch_normalization_33_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
6batch_normalization_33/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_33_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
8batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_33_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
'batch_normalization_33/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_33/MaxPool:output:0-batch_normalization_33/ReadVariableOp:value:0/batch_normalization_33/ReadVariableOp_1:value:0>batch_normalization_33/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
is_training( ?
conv2d_61/Conv2D/ReadVariableOpReadVariableOp(conv2d_61_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
conv2d_61/Conv2DConv2D+batch_normalization_33/FusedBatchNormV3:y:0'conv2d_61/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????
?*
paddingVALID*
strides
?
 conv2d_61/BiasAdd/ReadVariableOpReadVariableOp)conv2d_61_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
conv2d_61/BiasAddBiasAddconv2d_61/Conv2D:output:0(conv2d_61/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????
?m
conv2d_61/ReluReluconv2d_61/BiasAdd:output:0*
T0*0
_output_shapes
:?????????
??
conv2d_62/Conv2D/ReadVariableOpReadVariableOp(conv2d_62_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
conv2d_62/Conv2DConv2Dconv2d_61/Relu:activations:0'conv2d_62/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
?
 conv2d_62/BiasAdd/ReadVariableOpReadVariableOp)conv2d_62_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
conv2d_62/BiasAddBiasAddconv2d_62/Conv2D:output:0(conv2d_62/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????m
conv2d_62/ReluReluconv2d_62/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
max_pooling2d_34/MaxPoolMaxPoolconv2d_62/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
?
%batch_normalization_34/ReadVariableOpReadVariableOp.batch_normalization_34_readvariableop_resource*
_output_shapes	
:?*
dtype0?
'batch_normalization_34/ReadVariableOp_1ReadVariableOp0batch_normalization_34_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
6batch_normalization_34/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_34_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
8batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_34_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
'batch_normalization_34/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_34/MaxPool:output:0-batch_normalization_34/ReadVariableOp:value:0/batch_normalization_34/ReadVariableOp_1:value:0>batch_normalization_34/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( ?
conv2d_63/Conv2D/ReadVariableOpReadVariableOp(conv2d_63_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
conv2d_63/Conv2DConv2D+batch_normalization_34/FusedBatchNormV3:y:0'conv2d_63/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	?*
paddingVALID*
strides
?
 conv2d_63/BiasAdd/ReadVariableOpReadVariableOp)conv2d_63_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
conv2d_63/BiasAddBiasAddconv2d_63/Conv2D:output:0(conv2d_63/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	?m
conv2d_63/ReluReluconv2d_63/BiasAdd:output:0*
T0*0
_output_shapes
:?????????	??
max_pooling2d_35/MaxPoolMaxPoolconv2d_63/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
?
%batch_normalization_35/ReadVariableOpReadVariableOp.batch_normalization_35_readvariableop_resource*
_output_shapes	
:?*
dtype0?
'batch_normalization_35/ReadVariableOp_1ReadVariableOp0batch_normalization_35_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
6batch_normalization_35/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_35_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
8batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_35_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
'batch_normalization_35/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_35/MaxPool:output:0-batch_normalization_35/ReadVariableOp:value:0/batch_normalization_35/ReadVariableOp_1:value:0>batch_normalization_35/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( a
flatten_11/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ?
flatten_11/ReshapeReshape+batch_normalization_35/FusedBatchNormV3:y:0flatten_11/Const:output:0*
T0*(
_output_shapes
:???????????
dense_22/MatMul/ReadVariableOpReadVariableOp'dense_22_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense_22/MatMulMatMulflatten_11/Reshape:output:0&dense_22/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_22/BiasAdd/ReadVariableOpReadVariableOp(dense_22_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_22/BiasAddBiasAdddense_22/MatMul:product:0'dense_22/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????c
dense_22/ReluReludense_22/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
dense_23/MatMul/ReadVariableOpReadVariableOp'dense_23_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
dense_23/MatMulMatMuldense_22/Relu:activations:0&dense_23/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_23/BiasAdd/ReadVariableOpReadVariableOp(dense_23_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_23/BiasAddBiasAdddense_23/MatMul:product:0'dense_23/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????h
dense_23/SoftmaxSoftmaxdense_23/BiasAdd:output:0*
T0*'
_output_shapes
:?????????i
IdentityIdentitydense_23/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp7^batch_normalization_33/FusedBatchNormV3/ReadVariableOp9^batch_normalization_33/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_33/ReadVariableOp(^batch_normalization_33/ReadVariableOp_17^batch_normalization_34/FusedBatchNormV3/ReadVariableOp9^batch_normalization_34/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_34/ReadVariableOp(^batch_normalization_34/ReadVariableOp_17^batch_normalization_35/FusedBatchNormV3/ReadVariableOp9^batch_normalization_35/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_35/ReadVariableOp(^batch_normalization_35/ReadVariableOp_1!^conv2d_59/BiasAdd/ReadVariableOp ^conv2d_59/Conv2D/ReadVariableOp!^conv2d_60/BiasAdd/ReadVariableOp ^conv2d_60/Conv2D/ReadVariableOp!^conv2d_61/BiasAdd/ReadVariableOp ^conv2d_61/Conv2D/ReadVariableOp!^conv2d_62/BiasAdd/ReadVariableOp ^conv2d_62/Conv2D/ReadVariableOp!^conv2d_63/BiasAdd/ReadVariableOp ^conv2d_63/Conv2D/ReadVariableOp ^dense_22/BiasAdd/ReadVariableOp^dense_22/MatMul/ReadVariableOp ^dense_23/BiasAdd/ReadVariableOp^dense_23/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:?????????8: : : : : : : : : : : : : : : : : : : : : : : : : : 2p
6batch_normalization_33/FusedBatchNormV3/ReadVariableOp6batch_normalization_33/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_33/FusedBatchNormV3/ReadVariableOp_18batch_normalization_33/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_33/ReadVariableOp%batch_normalization_33/ReadVariableOp2R
'batch_normalization_33/ReadVariableOp_1'batch_normalization_33/ReadVariableOp_12p
6batch_normalization_34/FusedBatchNormV3/ReadVariableOp6batch_normalization_34/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_34/FusedBatchNormV3/ReadVariableOp_18batch_normalization_34/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_34/ReadVariableOp%batch_normalization_34/ReadVariableOp2R
'batch_normalization_34/ReadVariableOp_1'batch_normalization_34/ReadVariableOp_12p
6batch_normalization_35/FusedBatchNormV3/ReadVariableOp6batch_normalization_35/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_35/FusedBatchNormV3/ReadVariableOp_18batch_normalization_35/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_35/ReadVariableOp%batch_normalization_35/ReadVariableOp2R
'batch_normalization_35/ReadVariableOp_1'batch_normalization_35/ReadVariableOp_12D
 conv2d_59/BiasAdd/ReadVariableOp conv2d_59/BiasAdd/ReadVariableOp2B
conv2d_59/Conv2D/ReadVariableOpconv2d_59/Conv2D/ReadVariableOp2D
 conv2d_60/BiasAdd/ReadVariableOp conv2d_60/BiasAdd/ReadVariableOp2B
conv2d_60/Conv2D/ReadVariableOpconv2d_60/Conv2D/ReadVariableOp2D
 conv2d_61/BiasAdd/ReadVariableOp conv2d_61/BiasAdd/ReadVariableOp2B
conv2d_61/Conv2D/ReadVariableOpconv2d_61/Conv2D/ReadVariableOp2D
 conv2d_62/BiasAdd/ReadVariableOp conv2d_62/BiasAdd/ReadVariableOp2B
conv2d_62/Conv2D/ReadVariableOpconv2d_62/Conv2D/ReadVariableOp2D
 conv2d_63/BiasAdd/ReadVariableOp conv2d_63/BiasAdd/ReadVariableOp2B
conv2d_63/Conv2D/ReadVariableOpconv2d_63/Conv2D/ReadVariableOp2B
dense_22/BiasAdd/ReadVariableOpdense_22/BiasAdd/ReadVariableOp2@
dense_22/MatMul/ReadVariableOpdense_22/MatMul/ReadVariableOp2B
dense_23/BiasAdd/ReadVariableOpdense_23/BiasAdd/ReadVariableOp2@
dense_23/MatMul/ReadVariableOpdense_23/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????8
 
_user_specified_nameinputs
?
i
M__inference_max_pooling2d_35_layer_call_and_return_conditional_losses_2446043

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs"?	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
S
conv2d_59_input@
!serving_default_conv2d_59_input:0?????????8<
dense_230
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer-6
layer_with_weights-5
layer-7
	layer_with_weights-6
	layer-8

layer-9
layer_with_weights-7
layer-10
layer-11
layer_with_weights-8
layer-12
layer_with_weights-9
layer-13
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature
	optimizer

signatures"
_tf_keras_sequential
?
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
  _jit_compiled_convolution_op"
_tf_keras_layer
?
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses

'kernel
(bias
 )_jit_compiled_convolution_op"
_tf_keras_layer
?
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses"
_tf_keras_layer
?
0	variables
1trainable_variables
2regularization_losses
3	keras_api
4__call__
*5&call_and_return_all_conditional_losses
6axis
	7gamma
8beta
9moving_mean
:moving_variance"
_tf_keras_layer
?
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
*@&call_and_return_all_conditional_losses

Akernel
Bbias
 C_jit_compiled_convolution_op"
_tf_keras_layer
?
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
H__call__
*I&call_and_return_all_conditional_losses

Jkernel
Kbias
 L_jit_compiled_convolution_op"
_tf_keras_layer
?
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses"
_tf_keras_layer
?
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses
Yaxis
	Zgamma
[beta
\moving_mean
]moving_variance"
_tf_keras_layer
?
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses

dkernel
ebias
 f_jit_compiled_convolution_op"
_tf_keras_layer
?
g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses"
_tf_keras_layer
?
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
q__call__
*r&call_and_return_all_conditional_losses
saxis
	tgamma
ubeta
vmoving_mean
wmoving_variance"
_tf_keras_layer
?
x	variables
ytrainable_variables
zregularization_losses
{	keras_api
|__call__
*}&call_and_return_all_conditional_losses"
_tf_keras_layer
?
~	variables
trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias"
_tf_keras_layer
?
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses
?kernel
	?bias"
_tf_keras_layer
?
0
1
'2
(3
74
85
96
:7
A8
B9
J10
K11
Z12
[13
\14
]15
d16
e17
t18
u19
v20
w21
?22
?23
?24
?25"
trackable_list_wrapper
?
0
1
'2
(3
74
85
A6
B7
J8
K9
Z10
[11
d12
e13
t14
u15
?16
?17
?18
?19"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_1
?trace_2
?trace_32?
/__inference_sequential_13_layer_call_fn_2444953
/__inference_sequential_13_layer_call_fn_2445532
/__inference_sequential_13_layer_call_fn_2445589
/__inference_sequential_13_layer_call_fn_2445270?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1z?trace_2z?trace_3
?
?trace_0
?trace_1
?trace_2
?trace_32?
J__inference_sequential_13_layer_call_and_return_conditional_losses_2445689
J__inference_sequential_13_layer_call_and_return_conditional_losses_2445789
J__inference_sequential_13_layer_call_and_return_conditional_losses_2445340
J__inference_sequential_13_layer_call_and_return_conditional_losses_2445410?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1z?trace_2z?trace_3
?B?
"__inference__wrapped_model_2444505conv2d_59_input"?
???
FullArgSpec
args? 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?
	?iter
?beta_1
?beta_2

?decay
?learning_ratem?m?'m?(m?7m?8m?Am?Bm?Jm?Km?Zm?[m?dm?em?tm?um?	?m?	?m?	?m?	?m?v?v?'v?(v?7v?8v?Av?Bv?Jv?Kv?Zv?[v?dv?ev?tv?uv?	?v?	?v?	?v?	?v?"
	optimizer
-
?serving_default"
signature_map
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
+__inference_conv2d_59_layer_call_fn_2445798?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
F__inference_conv2d_59_layer_call_and_return_conditional_losses_2445809?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
*:(@2conv2d_59/kernel
:@2conv2d_59/bias
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
.
'0
(1"
trackable_list_wrapper
.
'0
(1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
+__inference_conv2d_60_layer_call_fn_2445818?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
F__inference_conv2d_60_layer_call_and_return_conditional_losses_2445829?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
*:(@@2conv2d_60/kernel
:@2conv2d_60/bias
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
2__inference_max_pooling2d_33_layer_call_fn_2445834?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
M__inference_max_pooling2d_33_layer_call_and_return_conditional_losses_2445839?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
<
70
81
92
:3"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
8__inference_batch_normalization_33_layer_call_fn_2445852
8__inference_batch_normalization_33_layer_call_fn_2445865?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
S__inference_batch_normalization_33_layer_call_and_return_conditional_losses_2445883
S__inference_batch_normalization_33_layer_call_and_return_conditional_losses_2445901?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1
 "
trackable_list_wrapper
*:(@2batch_normalization_33/gamma
):'@2batch_normalization_33/beta
2:0@ (2"batch_normalization_33/moving_mean
6:4@ (2&batch_normalization_33/moving_variance
.
A0
B1"
trackable_list_wrapper
.
A0
B1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
;	variables
<trainable_variables
=regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
+__inference_conv2d_61_layer_call_fn_2445910?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
F__inference_conv2d_61_layer_call_and_return_conditional_losses_2445921?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
+:)@?2conv2d_61/kernel
:?2conv2d_61/bias
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
.
J0
K1"
trackable_list_wrapper
.
J0
K1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
D	variables
Etrainable_variables
Fregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
+__inference_conv2d_62_layer_call_fn_2445930?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
F__inference_conv2d_62_layer_call_and_return_conditional_losses_2445941?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
,:*??2conv2d_62/kernel
:?2conv2d_62/bias
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
2__inference_max_pooling2d_34_layer_call_fn_2445946?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
M__inference_max_pooling2d_34_layer_call_and_return_conditional_losses_2445951?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
<
Z0
[1
\2
]3"
trackable_list_wrapper
.
Z0
[1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
8__inference_batch_normalization_34_layer_call_fn_2445964
8__inference_batch_normalization_34_layer_call_fn_2445977?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
S__inference_batch_normalization_34_layer_call_and_return_conditional_losses_2445995
S__inference_batch_normalization_34_layer_call_and_return_conditional_losses_2446013?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1
 "
trackable_list_wrapper
+:)?2batch_normalization_34/gamma
*:(?2batch_normalization_34/beta
3:1? (2"batch_normalization_34/moving_mean
7:5? (2&batch_normalization_34/moving_variance
.
d0
e1"
trackable_list_wrapper
.
d0
e1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
+__inference_conv2d_63_layer_call_fn_2446022?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
F__inference_conv2d_63_layer_call_and_return_conditional_losses_2446033?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
,:*??2conv2d_63/kernel
:?2conv2d_63/bias
?2??
???
FullArgSpec'
args?
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
2__inference_max_pooling2d_35_layer_call_fn_2446038?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
M__inference_max_pooling2d_35_layer_call_and_return_conditional_losses_2446043?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
<
t0
u1
v2
w3"
trackable_list_wrapper
.
t0
u1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
m	variables
ntrainable_variables
oregularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
?
?trace_0
?trace_12?
8__inference_batch_normalization_35_layer_call_fn_2446056
8__inference_batch_normalization_35_layer_call_fn_2446069?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1
?
?trace_0
?trace_12?
S__inference_batch_normalization_35_layer_call_and_return_conditional_losses_2446087
S__inference_batch_normalization_35_layer_call_and_return_conditional_losses_2446105?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0z?trace_1
 "
trackable_list_wrapper
+:)?2batch_normalization_35/gamma
*:(?2batch_normalization_35/beta
3:1? (2"batch_normalization_35/moving_mean
7:5? (2&batch_normalization_35/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
x	variables
ytrainable_variables
zregularization_losses
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
,__inference_flatten_11_layer_call_fn_2446110?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
G__inference_flatten_11_layer_call_and_return_conditional_losses_2446116?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
~	variables
trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
*__inference_dense_22_layer_call_fn_2446125?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
E__inference_dense_22_layer_call_and_return_conditional_losses_2446136?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
#:!
??2dense_22/kernel
:?2dense_22/bias
0
?0
?1"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
?layers
?metrics
 ?layer_regularization_losses
?layer_metrics
?	variables
?trainable_variables
?regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
?trace_02?
*__inference_dense_23_layer_call_fn_2446145?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
?
?trace_02?
E__inference_dense_23_layer_call_and_return_conditional_losses_2446156?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 z?trace_0
": 	?2dense_23/kernel
:2dense_23/bias
J
90
:1
\2
]3
v4
w5"
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13"
trackable_list_wrapper
0
?0
?1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
/__inference_sequential_13_layer_call_fn_2444953conv2d_59_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
/__inference_sequential_13_layer_call_fn_2445532inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
/__inference_sequential_13_layer_call_fn_2445589inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
/__inference_sequential_13_layer_call_fn_2445270conv2d_59_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
J__inference_sequential_13_layer_call_and_return_conditional_losses_2445689inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
J__inference_sequential_13_layer_call_and_return_conditional_losses_2445789inputs"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
J__inference_sequential_13_layer_call_and_return_conditional_losses_2445340conv2d_59_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
J__inference_sequential_13_layer_call_and_return_conditional_losses_2445410conv2d_59_input"?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
?B?
%__inference_signature_wrapper_2445475conv2d_59_input"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
+__inference_conv2d_59_layer_call_fn_2445798inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
F__inference_conv2d_59_layer_call_and_return_conditional_losses_2445809inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
+__inference_conv2d_60_layer_call_fn_2445818inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
F__inference_conv2d_60_layer_call_and_return_conditional_losses_2445829inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
2__inference_max_pooling2d_33_layer_call_fn_2445834inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
M__inference_max_pooling2d_33_layer_call_and_return_conditional_losses_2445839inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
8__inference_batch_normalization_33_layer_call_fn_2445852inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
8__inference_batch_normalization_33_layer_call_fn_2445865inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
S__inference_batch_normalization_33_layer_call_and_return_conditional_losses_2445883inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
S__inference_batch_normalization_33_layer_call_and_return_conditional_losses_2445901inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
+__inference_conv2d_61_layer_call_fn_2445910inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
F__inference_conv2d_61_layer_call_and_return_conditional_losses_2445921inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
+__inference_conv2d_62_layer_call_fn_2445930inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
F__inference_conv2d_62_layer_call_and_return_conditional_losses_2445941inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
2__inference_max_pooling2d_34_layer_call_fn_2445946inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
M__inference_max_pooling2d_34_layer_call_and_return_conditional_losses_2445951inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
\0
]1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
8__inference_batch_normalization_34_layer_call_fn_2445964inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
8__inference_batch_normalization_34_layer_call_fn_2445977inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
S__inference_batch_normalization_34_layer_call_and_return_conditional_losses_2445995inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
S__inference_batch_normalization_34_layer_call_and_return_conditional_losses_2446013inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
+__inference_conv2d_63_layer_call_fn_2446022inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
F__inference_conv2d_63_layer_call_and_return_conditional_losses_2446033inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
2__inference_max_pooling2d_35_layer_call_fn_2446038inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
M__inference_max_pooling2d_35_layer_call_and_return_conditional_losses_2446043inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
.
v0
w1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
8__inference_batch_normalization_35_layer_call_fn_2446056inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
8__inference_batch_normalization_35_layer_call_fn_2446069inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
S__inference_batch_normalization_35_layer_call_and_return_conditional_losses_2446087inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
S__inference_batch_normalization_35_layer_call_and_return_conditional_losses_2446105inputs"?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
,__inference_flatten_11_layer_call_fn_2446110inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
G__inference_flatten_11_layer_call_and_return_conditional_losses_2446116inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
*__inference_dense_22_layer_call_fn_2446125inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
E__inference_dense_22_layer_call_and_return_conditional_losses_2446136inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?B?
*__inference_dense_23_layer_call_fn_2446145inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
E__inference_dense_23_layer_call_and_return_conditional_losses_2446156inputs"?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
R
?	variables
?	keras_api

?total

?count"
_tf_keras_metric
c
?	variables
?	keras_api

?total

?count
?
_fn_kwargs"
_tf_keras_metric
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
/:-@2Adam/conv2d_59/kernel/m
!:@2Adam/conv2d_59/bias/m
/:-@@2Adam/conv2d_60/kernel/m
!:@2Adam/conv2d_60/bias/m
/:-@2#Adam/batch_normalization_33/gamma/m
.:,@2"Adam/batch_normalization_33/beta/m
0:.@?2Adam/conv2d_61/kernel/m
": ?2Adam/conv2d_61/bias/m
1:/??2Adam/conv2d_62/kernel/m
": ?2Adam/conv2d_62/bias/m
0:.?2#Adam/batch_normalization_34/gamma/m
/:-?2"Adam/batch_normalization_34/beta/m
1:/??2Adam/conv2d_63/kernel/m
": ?2Adam/conv2d_63/bias/m
0:.?2#Adam/batch_normalization_35/gamma/m
/:-?2"Adam/batch_normalization_35/beta/m
(:&
??2Adam/dense_22/kernel/m
!:?2Adam/dense_22/bias/m
':%	?2Adam/dense_23/kernel/m
 :2Adam/dense_23/bias/m
/:-@2Adam/conv2d_59/kernel/v
!:@2Adam/conv2d_59/bias/v
/:-@@2Adam/conv2d_60/kernel/v
!:@2Adam/conv2d_60/bias/v
/:-@2#Adam/batch_normalization_33/gamma/v
.:,@2"Adam/batch_normalization_33/beta/v
0:.@?2Adam/conv2d_61/kernel/v
": ?2Adam/conv2d_61/bias/v
1:/??2Adam/conv2d_62/kernel/v
": ?2Adam/conv2d_62/bias/v
0:.?2#Adam/batch_normalization_34/gamma/v
/:-?2"Adam/batch_normalization_34/beta/v
1:/??2Adam/conv2d_63/kernel/v
": ?2Adam/conv2d_63/bias/v
0:.?2#Adam/batch_normalization_35/gamma/v
/:-?2"Adam/batch_normalization_35/beta/v
(:&
??2Adam/dense_22/kernel/v
!:?2Adam/dense_22/bias/v
':%	?2Adam/dense_23/kernel/v
 :2Adam/dense_23/bias/v?
"__inference__wrapped_model_2444505?'(789:ABJKZ[\]detuvw????@?=
6?3
1?.
conv2d_59_input?????????8
? "3?0
.
dense_23"?
dense_23??????????
S__inference_batch_normalization_33_layer_call_and_return_conditional_losses_2445883?789:M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
S__inference_batch_normalization_33_layer_call_and_return_conditional_losses_2445901?789:M?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
8__inference_batch_normalization_33_layer_call_fn_2445852?789:M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
8__inference_batch_normalization_33_layer_call_fn_2445865?789:M?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
S__inference_batch_normalization_34_layer_call_and_return_conditional_losses_2445995?Z[\]N?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
S__inference_batch_normalization_34_layer_call_and_return_conditional_losses_2446013?Z[\]N?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
8__inference_batch_normalization_34_layer_call_fn_2445964?Z[\]N?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
8__inference_batch_normalization_34_layer_call_fn_2445977?Z[\]N?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
S__inference_batch_normalization_35_layer_call_and_return_conditional_losses_2446087?tuvwN?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
S__inference_batch_normalization_35_layer_call_and_return_conditional_losses_2446105?tuvwN?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
8__inference_batch_normalization_35_layer_call_fn_2446056?tuvwN?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
8__inference_batch_normalization_35_layer_call_fn_2446069?tuvwN?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
F__inference_conv2d_59_layer_call_and_return_conditional_losses_2445809l7?4
-?*
(?%
inputs?????????8
? "-?*
#? 
0?????????6@
? ?
+__inference_conv2d_59_layer_call_fn_2445798_7?4
-?*
(?%
inputs?????????8
? " ??????????6@?
F__inference_conv2d_60_layer_call_and_return_conditional_losses_2445829l'(7?4
-?*
(?%
inputs?????????6@
? "-?*
#? 
0?????????4@
? ?
+__inference_conv2d_60_layer_call_fn_2445818_'(7?4
-?*
(?%
inputs?????????6@
? " ??????????4@?
F__inference_conv2d_61_layer_call_and_return_conditional_losses_2445921mAB7?4
-?*
(?%
inputs?????????@
? ".?+
$?!
0?????????
?
? ?
+__inference_conv2d_61_layer_call_fn_2445910`AB7?4
-?*
(?%
inputs?????????@
? "!??????????
??
F__inference_conv2d_62_layer_call_and_return_conditional_losses_2445941nJK8?5
.?+
)?&
inputs?????????
?
? ".?+
$?!
0??????????
? ?
+__inference_conv2d_62_layer_call_fn_2445930aJK8?5
.?+
)?&
inputs?????????
?
? "!????????????
F__inference_conv2d_63_layer_call_and_return_conditional_losses_2446033nde8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0?????????	?
? ?
+__inference_conv2d_63_layer_call_fn_2446022ade8?5
.?+
)?&
inputs??????????
? "!??????????	??
E__inference_dense_22_layer_call_and_return_conditional_losses_2446136`??0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
*__inference_dense_22_layer_call_fn_2446125S??0?-
&?#
!?
inputs??????????
? "????????????
E__inference_dense_23_layer_call_and_return_conditional_losses_2446156_??0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? ?
*__inference_dense_23_layer_call_fn_2446145R??0?-
&?#
!?
inputs??????????
? "???????????
G__inference_flatten_11_layer_call_and_return_conditional_losses_2446116b8?5
.?+
)?&
inputs??????????
? "&?#
?
0??????????
? ?
,__inference_flatten_11_layer_call_fn_2446110U8?5
.?+
)?&
inputs??????????
? "????????????
M__inference_max_pooling2d_33_layer_call_and_return_conditional_losses_2445839?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
2__inference_max_pooling2d_33_layer_call_fn_2445834?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
M__inference_max_pooling2d_34_layer_call_and_return_conditional_losses_2445951?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
2__inference_max_pooling2d_34_layer_call_fn_2445946?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
M__inference_max_pooling2d_35_layer_call_and_return_conditional_losses_2446043?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
2__inference_max_pooling2d_35_layer_call_fn_2446038?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
J__inference_sequential_13_layer_call_and_return_conditional_losses_2445340?'(789:ABJKZ[\]detuvw????H?E
>?;
1?.
conv2d_59_input?????????8
p 

 
? "%?"
?
0?????????
? ?
J__inference_sequential_13_layer_call_and_return_conditional_losses_2445410?'(789:ABJKZ[\]detuvw????H?E
>?;
1?.
conv2d_59_input?????????8
p

 
? "%?"
?
0?????????
? ?
J__inference_sequential_13_layer_call_and_return_conditional_losses_2445689?'(789:ABJKZ[\]detuvw??????<
5?2
(?%
inputs?????????8
p 

 
? "%?"
?
0?????????
? ?
J__inference_sequential_13_layer_call_and_return_conditional_losses_2445789?'(789:ABJKZ[\]detuvw??????<
5?2
(?%
inputs?????????8
p

 
? "%?"
?
0?????????
? ?
/__inference_sequential_13_layer_call_fn_2444953?'(789:ABJKZ[\]detuvw????H?E
>?;
1?.
conv2d_59_input?????????8
p 

 
? "???????????
/__inference_sequential_13_layer_call_fn_2445270?'(789:ABJKZ[\]detuvw????H?E
>?;
1?.
conv2d_59_input?????????8
p

 
? "???????????
/__inference_sequential_13_layer_call_fn_2445532{'(789:ABJKZ[\]detuvw??????<
5?2
(?%
inputs?????????8
p 

 
? "???????????
/__inference_sequential_13_layer_call_fn_2445589{'(789:ABJKZ[\]detuvw??????<
5?2
(?%
inputs?????????8
p

 
? "???????????
%__inference_signature_wrapper_2445475?'(789:ABJKZ[\]detuvw????S?P
? 
I?F
D
conv2d_59_input1?.
conv2d_59_input?????????8"3?0
.
dense_23"?
dense_23?????????