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
Adam/dense_27/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_27/bias/v
y
(Adam/dense_27/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_27/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense_27/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_27/kernel/v
?
*Adam/dense_27/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_27/kernel/v*
_output_shapes
:	?*
dtype0
?
Adam/dense_26/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_26/bias/v
z
(Adam/dense_26/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_26/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/dense_26/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_26/kernel/v
?
*Adam/dense_26/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_26/kernel/v* 
_output_shapes
:
??*
dtype0
?
"Adam/batch_normalization_41/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Adam/batch_normalization_41/beta/v
?
6Adam/batch_normalization_41/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_41/beta/v*
_output_shapes	
:?*
dtype0
?
#Adam/batch_normalization_41/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#Adam/batch_normalization_41/gamma/v
?
7Adam/batch_normalization_41/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_41/gamma/v*
_output_shapes	
:?*
dtype0
?
Adam/conv2d_73/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/conv2d_73/bias/v
|
)Adam/conv2d_73/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_73/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/conv2d_73/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*(
shared_nameAdam/conv2d_73/kernel/v
?
+Adam/conv2d_73/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_73/kernel/v*(
_output_shapes
:??*
dtype0
?
"Adam/batch_normalization_40/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Adam/batch_normalization_40/beta/v
?
6Adam/batch_normalization_40/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_40/beta/v*
_output_shapes	
:?*
dtype0
?
#Adam/batch_normalization_40/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#Adam/batch_normalization_40/gamma/v
?
7Adam/batch_normalization_40/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_40/gamma/v*
_output_shapes	
:?*
dtype0
?
Adam/conv2d_72/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/conv2d_72/bias/v
|
)Adam/conv2d_72/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_72/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/conv2d_72/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*(
shared_nameAdam/conv2d_72/kernel/v
?
+Adam/conv2d_72/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_72/kernel/v*(
_output_shapes
:??*
dtype0
?
Adam/conv2d_71/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/conv2d_71/bias/v
|
)Adam/conv2d_71/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_71/bias/v*
_output_shapes	
:?*
dtype0
?
Adam/conv2d_71/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*(
shared_nameAdam/conv2d_71/kernel/v
?
+Adam/conv2d_71/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_71/kernel/v*'
_output_shapes
:@?*
dtype0
?
"Adam/batch_normalization_39/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_39/beta/v
?
6Adam/batch_normalization_39/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_39/beta/v*
_output_shapes
:@*
dtype0
?
#Adam/batch_normalization_39/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/batch_normalization_39/gamma/v
?
7Adam/batch_normalization_39/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_39/gamma/v*
_output_shapes
:@*
dtype0
?
Adam/conv2d_70/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_70/bias/v
{
)Adam/conv2d_70/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_70/bias/v*
_output_shapes
:@*
dtype0
?
Adam/conv2d_70/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv2d_70/kernel/v
?
+Adam/conv2d_70/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_70/kernel/v*&
_output_shapes
:@@*
dtype0
?
Adam/conv2d_69/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_69/bias/v
{
)Adam/conv2d_69/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_69/bias/v*
_output_shapes
:@*
dtype0
?
Adam/conv2d_69/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/conv2d_69/kernel/v
?
+Adam/conv2d_69/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_69/kernel/v*&
_output_shapes
:@*
dtype0
?
Adam/dense_27/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_27/bias/m
y
(Adam/dense_27/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_27/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense_27/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*'
shared_nameAdam/dense_27/kernel/m
?
*Adam/dense_27/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_27/kernel/m*
_output_shapes
:	?*
dtype0
?
Adam/dense_26/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*%
shared_nameAdam/dense_26/bias/m
z
(Adam/dense_26/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_26/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/dense_26/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??*'
shared_nameAdam/dense_26/kernel/m
?
*Adam/dense_26/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_26/kernel/m* 
_output_shapes
:
??*
dtype0
?
"Adam/batch_normalization_41/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Adam/batch_normalization_41/beta/m
?
6Adam/batch_normalization_41/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_41/beta/m*
_output_shapes	
:?*
dtype0
?
#Adam/batch_normalization_41/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#Adam/batch_normalization_41/gamma/m
?
7Adam/batch_normalization_41/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_41/gamma/m*
_output_shapes	
:?*
dtype0
?
Adam/conv2d_73/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/conv2d_73/bias/m
|
)Adam/conv2d_73/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_73/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/conv2d_73/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*(
shared_nameAdam/conv2d_73/kernel/m
?
+Adam/conv2d_73/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_73/kernel/m*(
_output_shapes
:??*
dtype0
?
"Adam/batch_normalization_40/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"Adam/batch_normalization_40/beta/m
?
6Adam/batch_normalization_40/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_40/beta/m*
_output_shapes	
:?*
dtype0
?
#Adam/batch_normalization_40/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*4
shared_name%#Adam/batch_normalization_40/gamma/m
?
7Adam/batch_normalization_40/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_40/gamma/m*
_output_shapes	
:?*
dtype0
?
Adam/conv2d_72/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/conv2d_72/bias/m
|
)Adam/conv2d_72/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_72/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/conv2d_72/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*(
shared_nameAdam/conv2d_72/kernel/m
?
+Adam/conv2d_72/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_72/kernel/m*(
_output_shapes
:??*
dtype0
?
Adam/conv2d_71/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameAdam/conv2d_71/bias/m
|
)Adam/conv2d_71/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_71/bias/m*
_output_shapes	
:?*
dtype0
?
Adam/conv2d_71/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*(
shared_nameAdam/conv2d_71/kernel/m
?
+Adam/conv2d_71/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_71/kernel/m*'
_output_shapes
:@?*
dtype0
?
"Adam/batch_normalization_39/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_39/beta/m
?
6Adam/batch_normalization_39/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_39/beta/m*
_output_shapes
:@*
dtype0
?
#Adam/batch_normalization_39/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/batch_normalization_39/gamma/m
?
7Adam/batch_normalization_39/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_39/gamma/m*
_output_shapes
:@*
dtype0
?
Adam/conv2d_70/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_70/bias/m
{
)Adam/conv2d_70/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_70/bias/m*
_output_shapes
:@*
dtype0
?
Adam/conv2d_70/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv2d_70/kernel/m
?
+Adam/conv2d_70/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_70/kernel/m*&
_output_shapes
:@@*
dtype0
?
Adam/conv2d_69/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_69/bias/m
{
)Adam/conv2d_69/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_69/bias/m*
_output_shapes
:@*
dtype0
?
Adam/conv2d_69/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/conv2d_69/kernel/m
?
+Adam/conv2d_69/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_69/kernel/m*&
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
dense_27/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_27/bias
k
!dense_27/bias/Read/ReadVariableOpReadVariableOpdense_27/bias*
_output_shapes
:*
dtype0
{
dense_27/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?* 
shared_namedense_27/kernel
t
#dense_27/kernel/Read/ReadVariableOpReadVariableOpdense_27/kernel*
_output_shapes
:	?*
dtype0
s
dense_26/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namedense_26/bias
l
!dense_26/bias/Read/ReadVariableOpReadVariableOpdense_26/bias*
_output_shapes	
:?*
dtype0
|
dense_26/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??* 
shared_namedense_26/kernel
u
#dense_26/kernel/Read/ReadVariableOpReadVariableOpdense_26/kernel* 
_output_shapes
:
??*
dtype0
?
&batch_normalization_41/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&batch_normalization_41/moving_variance
?
:batch_normalization_41/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_41/moving_variance*
_output_shapes	
:?*
dtype0
?
"batch_normalization_41/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"batch_normalization_41/moving_mean
?
6batch_normalization_41/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_41/moving_mean*
_output_shapes	
:?*
dtype0
?
batch_normalization_41/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namebatch_normalization_41/beta
?
/batch_normalization_41/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_41/beta*
_output_shapes	
:?*
dtype0
?
batch_normalization_41/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_namebatch_normalization_41/gamma
?
0batch_normalization_41/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_41/gamma*
_output_shapes	
:?*
dtype0
u
conv2d_73/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d_73/bias
n
"conv2d_73/bias/Read/ReadVariableOpReadVariableOpconv2d_73/bias*
_output_shapes	
:?*
dtype0
?
conv2d_73/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*!
shared_nameconv2d_73/kernel

$conv2d_73/kernel/Read/ReadVariableOpReadVariableOpconv2d_73/kernel*(
_output_shapes
:??*
dtype0
?
&batch_normalization_40/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*7
shared_name(&batch_normalization_40/moving_variance
?
:batch_normalization_40/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_40/moving_variance*
_output_shapes	
:?*
dtype0
?
"batch_normalization_40/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*3
shared_name$"batch_normalization_40/moving_mean
?
6batch_normalization_40/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_40/moving_mean*
_output_shapes	
:?*
dtype0
?
batch_normalization_40/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*,
shared_namebatch_normalization_40/beta
?
/batch_normalization_40/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_40/beta*
_output_shapes	
:?*
dtype0
?
batch_normalization_40/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*-
shared_namebatch_normalization_40/gamma
?
0batch_normalization_40/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_40/gamma*
_output_shapes	
:?*
dtype0
u
conv2d_72/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d_72/bias
n
"conv2d_72/bias/Read/ReadVariableOpReadVariableOpconv2d_72/bias*
_output_shapes	
:?*
dtype0
?
conv2d_72/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*!
shared_nameconv2d_72/kernel

$conv2d_72/kernel/Read/ReadVariableOpReadVariableOpconv2d_72/kernel*(
_output_shapes
:??*
dtype0
u
conv2d_71/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d_71/bias
n
"conv2d_71/bias/Read/ReadVariableOpReadVariableOpconv2d_71/bias*
_output_shapes	
:?*
dtype0
?
conv2d_71/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@?*!
shared_nameconv2d_71/kernel
~
$conv2d_71/kernel/Read/ReadVariableOpReadVariableOpconv2d_71/kernel*'
_output_shapes
:@?*
dtype0
?
&batch_normalization_39/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_39/moving_variance
?
:batch_normalization_39/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_39/moving_variance*
_output_shapes
:@*
dtype0
?
"batch_normalization_39/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_39/moving_mean
?
6batch_normalization_39/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_39/moving_mean*
_output_shapes
:@*
dtype0
?
batch_normalization_39/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_39/beta
?
/batch_normalization_39/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_39/beta*
_output_shapes
:@*
dtype0
?
batch_normalization_39/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_39/gamma
?
0batch_normalization_39/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_39/gamma*
_output_shapes
:@*
dtype0
t
conv2d_70/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_70/bias
m
"conv2d_70/bias/Read/ReadVariableOpReadVariableOpconv2d_70/bias*
_output_shapes
:@*
dtype0
?
conv2d_70/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameconv2d_70/kernel
}
$conv2d_70/kernel/Read/ReadVariableOpReadVariableOpconv2d_70/kernel*&
_output_shapes
:@@*
dtype0
t
conv2d_69/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_69/bias
m
"conv2d_69/bias/Read/ReadVariableOpReadVariableOpconv2d_69/bias*
_output_shapes
:@*
dtype0
?
conv2d_69/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameconv2d_69/kernel
}
$conv2d_69/kernel/Read/ReadVariableOpReadVariableOpconv2d_69/kernel*&
_output_shapes
:@*
dtype0
?
serving_default_conv2d_69_inputPlaceholder*/
_output_shapes
:?????????8*
dtype0*$
shape:?????????8
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_69_inputconv2d_69/kernelconv2d_69/biasconv2d_70/kernelconv2d_70/biasbatch_normalization_39/gammabatch_normalization_39/beta"batch_normalization_39/moving_mean&batch_normalization_39/moving_varianceconv2d_71/kernelconv2d_71/biasconv2d_72/kernelconv2d_72/biasbatch_normalization_40/gammabatch_normalization_40/beta"batch_normalization_40/moving_mean&batch_normalization_40/moving_varianceconv2d_73/kernelconv2d_73/biasbatch_normalization_41/gammabatch_normalization_41/beta"batch_normalization_41/moving_mean&batch_normalization_41/moving_variancedense_26/kerneldense_26/biasdense_27/kerneldense_27/bias*&
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
%__inference_signature_wrapper_2450587

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
VARIABLE_VALUEconv2d_69/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_69/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEconv2d_70/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_70/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEbatch_normalization_39/gamma5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_39/beta4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_39/moving_mean;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_39/moving_variance?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEconv2d_71/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_71/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEconv2d_72/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_72/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEbatch_normalization_40/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_40/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_40/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_40/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEconv2d_73/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_73/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEbatch_normalization_41/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_41/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_41/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_41/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEdense_26/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_26/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEdense_27/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_27/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEAdam/conv2d_69/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_69/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/conv2d_70/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_70/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_39/gamma/mQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/batch_normalization_39/beta/mPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/conv2d_71/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_71/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/conv2d_72/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_72/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_40/gamma/mQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/batch_normalization_40/beta/mPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/conv2d_73/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_73/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_41/gamma/mQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/batch_normalization_41/beta/mPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEAdam/dense_26/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_26/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEAdam/dense_27/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_27/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/conv2d_69/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_69/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/conv2d_70/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_70/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_39/gamma/vQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/batch_normalization_39/beta/vPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/conv2d_71/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_71/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/conv2d_72/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_72/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_40/gamma/vQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/batch_normalization_40/beta/vPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?}
VARIABLE_VALUEAdam/conv2d_73/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_73/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE#Adam/batch_normalization_41/gamma/vQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
??
VARIABLE_VALUE"Adam/batch_normalization_41/beta/vPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEAdam/dense_26/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_26/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
?|
VARIABLE_VALUEAdam/dense_27/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_27/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_69/kernel/Read/ReadVariableOp"conv2d_69/bias/Read/ReadVariableOp$conv2d_70/kernel/Read/ReadVariableOp"conv2d_70/bias/Read/ReadVariableOp0batch_normalization_39/gamma/Read/ReadVariableOp/batch_normalization_39/beta/Read/ReadVariableOp6batch_normalization_39/moving_mean/Read/ReadVariableOp:batch_normalization_39/moving_variance/Read/ReadVariableOp$conv2d_71/kernel/Read/ReadVariableOp"conv2d_71/bias/Read/ReadVariableOp$conv2d_72/kernel/Read/ReadVariableOp"conv2d_72/bias/Read/ReadVariableOp0batch_normalization_40/gamma/Read/ReadVariableOp/batch_normalization_40/beta/Read/ReadVariableOp6batch_normalization_40/moving_mean/Read/ReadVariableOp:batch_normalization_40/moving_variance/Read/ReadVariableOp$conv2d_73/kernel/Read/ReadVariableOp"conv2d_73/bias/Read/ReadVariableOp0batch_normalization_41/gamma/Read/ReadVariableOp/batch_normalization_41/beta/Read/ReadVariableOp6batch_normalization_41/moving_mean/Read/ReadVariableOp:batch_normalization_41/moving_variance/Read/ReadVariableOp#dense_26/kernel/Read/ReadVariableOp!dense_26/bias/Read/ReadVariableOp#dense_27/kernel/Read/ReadVariableOp!dense_27/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp+Adam/conv2d_69/kernel/m/Read/ReadVariableOp)Adam/conv2d_69/bias/m/Read/ReadVariableOp+Adam/conv2d_70/kernel/m/Read/ReadVariableOp)Adam/conv2d_70/bias/m/Read/ReadVariableOp7Adam/batch_normalization_39/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_39/beta/m/Read/ReadVariableOp+Adam/conv2d_71/kernel/m/Read/ReadVariableOp)Adam/conv2d_71/bias/m/Read/ReadVariableOp+Adam/conv2d_72/kernel/m/Read/ReadVariableOp)Adam/conv2d_72/bias/m/Read/ReadVariableOp7Adam/batch_normalization_40/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_40/beta/m/Read/ReadVariableOp+Adam/conv2d_73/kernel/m/Read/ReadVariableOp)Adam/conv2d_73/bias/m/Read/ReadVariableOp7Adam/batch_normalization_41/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_41/beta/m/Read/ReadVariableOp*Adam/dense_26/kernel/m/Read/ReadVariableOp(Adam/dense_26/bias/m/Read/ReadVariableOp*Adam/dense_27/kernel/m/Read/ReadVariableOp(Adam/dense_27/bias/m/Read/ReadVariableOp+Adam/conv2d_69/kernel/v/Read/ReadVariableOp)Adam/conv2d_69/bias/v/Read/ReadVariableOp+Adam/conv2d_70/kernel/v/Read/ReadVariableOp)Adam/conv2d_70/bias/v/Read/ReadVariableOp7Adam/batch_normalization_39/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_39/beta/v/Read/ReadVariableOp+Adam/conv2d_71/kernel/v/Read/ReadVariableOp)Adam/conv2d_71/bias/v/Read/ReadVariableOp+Adam/conv2d_72/kernel/v/Read/ReadVariableOp)Adam/conv2d_72/bias/v/Read/ReadVariableOp7Adam/batch_normalization_40/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_40/beta/v/Read/ReadVariableOp+Adam/conv2d_73/kernel/v/Read/ReadVariableOp)Adam/conv2d_73/bias/v/Read/ReadVariableOp7Adam/batch_normalization_41/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_41/beta/v/Read/ReadVariableOp*Adam/dense_26/kernel/v/Read/ReadVariableOp(Adam/dense_26/bias/v/Read/ReadVariableOp*Adam/dense_27/kernel/v/Read/ReadVariableOp(Adam/dense_27/bias/v/Read/ReadVariableOpConst*X
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
 __inference__traced_save_2451516
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_69/kernelconv2d_69/biasconv2d_70/kernelconv2d_70/biasbatch_normalization_39/gammabatch_normalization_39/beta"batch_normalization_39/moving_mean&batch_normalization_39/moving_varianceconv2d_71/kernelconv2d_71/biasconv2d_72/kernelconv2d_72/biasbatch_normalization_40/gammabatch_normalization_40/beta"batch_normalization_40/moving_mean&batch_normalization_40/moving_varianceconv2d_73/kernelconv2d_73/biasbatch_normalization_41/gammabatch_normalization_41/beta"batch_normalization_41/moving_mean&batch_normalization_41/moving_variancedense_26/kerneldense_26/biasdense_27/kerneldense_27/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcountAdam/conv2d_69/kernel/mAdam/conv2d_69/bias/mAdam/conv2d_70/kernel/mAdam/conv2d_70/bias/m#Adam/batch_normalization_39/gamma/m"Adam/batch_normalization_39/beta/mAdam/conv2d_71/kernel/mAdam/conv2d_71/bias/mAdam/conv2d_72/kernel/mAdam/conv2d_72/bias/m#Adam/batch_normalization_40/gamma/m"Adam/batch_normalization_40/beta/mAdam/conv2d_73/kernel/mAdam/conv2d_73/bias/m#Adam/batch_normalization_41/gamma/m"Adam/batch_normalization_41/beta/mAdam/dense_26/kernel/mAdam/dense_26/bias/mAdam/dense_27/kernel/mAdam/dense_27/bias/mAdam/conv2d_69/kernel/vAdam/conv2d_69/bias/vAdam/conv2d_70/kernel/vAdam/conv2d_70/bias/v#Adam/batch_normalization_39/gamma/v"Adam/batch_normalization_39/beta/vAdam/conv2d_71/kernel/vAdam/conv2d_71/bias/vAdam/conv2d_72/kernel/vAdam/conv2d_72/bias/v#Adam/batch_normalization_40/gamma/v"Adam/batch_normalization_40/beta/vAdam/conv2d_73/kernel/vAdam/conv2d_73/bias/v#Adam/batch_normalization_41/gamma/v"Adam/batch_normalization_41/beta/vAdam/dense_26/kernel/vAdam/dense_26/bias/vAdam/dense_27/kernel/vAdam/dense_27/bias/v*W
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
#__inference__traced_restore_2451751??
?

?
E__inference_dense_27_layer_call_and_return_conditional_losses_2450003

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
?
?
/__inference_sequential_15_layer_call_fn_2450065
conv2d_69_input!
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
StatefulPartitionedCallStatefulPartitionedCallconv2d_69_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
J__inference_sequential_15_layer_call_and_return_conditional_losses_2450010o
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
_user_specified_nameconv2d_69_input
?	
?
8__inference_batch_normalization_41_layer_call_fn_2451168

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
S__inference_batch_normalization_41_layer_call_and_return_conditional_losses_2449803?
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
+__inference_conv2d_69_layer_call_fn_2450910

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
F__inference_conv2d_69_layer_call_and_return_conditional_losses_2449863w
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
?
?
S__inference_batch_normalization_40_layer_call_and_return_conditional_losses_2451107

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
+__inference_conv2d_71_layer_call_fn_2451022

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
F__inference_conv2d_71_layer_call_and_return_conditional_losses_2449907x
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
#__inference__traced_restore_2451751
file_prefix;
!assignvariableop_conv2d_69_kernel:@/
!assignvariableop_1_conv2d_69_bias:@=
#assignvariableop_2_conv2d_70_kernel:@@/
!assignvariableop_3_conv2d_70_bias:@=
/assignvariableop_4_batch_normalization_39_gamma:@<
.assignvariableop_5_batch_normalization_39_beta:@C
5assignvariableop_6_batch_normalization_39_moving_mean:@G
9assignvariableop_7_batch_normalization_39_moving_variance:@>
#assignvariableop_8_conv2d_71_kernel:@?0
!assignvariableop_9_conv2d_71_bias:	?@
$assignvariableop_10_conv2d_72_kernel:??1
"assignvariableop_11_conv2d_72_bias:	??
0assignvariableop_12_batch_normalization_40_gamma:	?>
/assignvariableop_13_batch_normalization_40_beta:	?E
6assignvariableop_14_batch_normalization_40_moving_mean:	?I
:assignvariableop_15_batch_normalization_40_moving_variance:	?@
$assignvariableop_16_conv2d_73_kernel:??1
"assignvariableop_17_conv2d_73_bias:	??
0assignvariableop_18_batch_normalization_41_gamma:	?>
/assignvariableop_19_batch_normalization_41_beta:	?E
6assignvariableop_20_batch_normalization_41_moving_mean:	?I
:assignvariableop_21_batch_normalization_41_moving_variance:	?7
#assignvariableop_22_dense_26_kernel:
??0
!assignvariableop_23_dense_26_bias:	?6
#assignvariableop_24_dense_27_kernel:	?/
!assignvariableop_25_dense_27_bias:'
assignvariableop_26_adam_iter:	 )
assignvariableop_27_adam_beta_1: )
assignvariableop_28_adam_beta_2: (
assignvariableop_29_adam_decay: 0
&assignvariableop_30_adam_learning_rate: %
assignvariableop_31_total_1: %
assignvariableop_32_count_1: #
assignvariableop_33_total: #
assignvariableop_34_count: E
+assignvariableop_35_adam_conv2d_69_kernel_m:@7
)assignvariableop_36_adam_conv2d_69_bias_m:@E
+assignvariableop_37_adam_conv2d_70_kernel_m:@@7
)assignvariableop_38_adam_conv2d_70_bias_m:@E
7assignvariableop_39_adam_batch_normalization_39_gamma_m:@D
6assignvariableop_40_adam_batch_normalization_39_beta_m:@F
+assignvariableop_41_adam_conv2d_71_kernel_m:@?8
)assignvariableop_42_adam_conv2d_71_bias_m:	?G
+assignvariableop_43_adam_conv2d_72_kernel_m:??8
)assignvariableop_44_adam_conv2d_72_bias_m:	?F
7assignvariableop_45_adam_batch_normalization_40_gamma_m:	?E
6assignvariableop_46_adam_batch_normalization_40_beta_m:	?G
+assignvariableop_47_adam_conv2d_73_kernel_m:??8
)assignvariableop_48_adam_conv2d_73_bias_m:	?F
7assignvariableop_49_adam_batch_normalization_41_gamma_m:	?E
6assignvariableop_50_adam_batch_normalization_41_beta_m:	?>
*assignvariableop_51_adam_dense_26_kernel_m:
??7
(assignvariableop_52_adam_dense_26_bias_m:	?=
*assignvariableop_53_adam_dense_27_kernel_m:	?6
(assignvariableop_54_adam_dense_27_bias_m:E
+assignvariableop_55_adam_conv2d_69_kernel_v:@7
)assignvariableop_56_adam_conv2d_69_bias_v:@E
+assignvariableop_57_adam_conv2d_70_kernel_v:@@7
)assignvariableop_58_adam_conv2d_70_bias_v:@E
7assignvariableop_59_adam_batch_normalization_39_gamma_v:@D
6assignvariableop_60_adam_batch_normalization_39_beta_v:@F
+assignvariableop_61_adam_conv2d_71_kernel_v:@?8
)assignvariableop_62_adam_conv2d_71_bias_v:	?G
+assignvariableop_63_adam_conv2d_72_kernel_v:??8
)assignvariableop_64_adam_conv2d_72_bias_v:	?F
7assignvariableop_65_adam_batch_normalization_40_gamma_v:	?E
6assignvariableop_66_adam_batch_normalization_40_beta_v:	?G
+assignvariableop_67_adam_conv2d_73_kernel_v:??8
)assignvariableop_68_adam_conv2d_73_bias_v:	?F
7assignvariableop_69_adam_batch_normalization_41_gamma_v:	?E
6assignvariableop_70_adam_batch_normalization_41_beta_v:	?>
*assignvariableop_71_adam_dense_26_kernel_v:
??7
(assignvariableop_72_adam_dense_26_bias_v:	?=
*assignvariableop_73_adam_dense_27_kernel_v:	?6
(assignvariableop_74_adam_dense_27_bias_v:
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
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_69_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_69_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_70_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_70_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_4AssignVariableOp/assignvariableop_4_batch_normalization_39_gammaIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_5AssignVariableOp.assignvariableop_5_batch_normalization_39_betaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_6AssignVariableOp5assignvariableop_6_batch_normalization_39_moving_meanIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_7AssignVariableOp9assignvariableop_7_batch_normalization_39_moving_varianceIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_8AssignVariableOp#assignvariableop_8_conv2d_71_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_9AssignVariableOp!assignvariableop_9_conv2d_71_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_10AssignVariableOp$assignvariableop_10_conv2d_72_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_11AssignVariableOp"assignvariableop_11_conv2d_72_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_12AssignVariableOp0assignvariableop_12_batch_normalization_40_gammaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_13AssignVariableOp/assignvariableop_13_batch_normalization_40_betaIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_14AssignVariableOp6assignvariableop_14_batch_normalization_40_moving_meanIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp:assignvariableop_15_batch_normalization_40_moving_varianceIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_16AssignVariableOp$assignvariableop_16_conv2d_73_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp"assignvariableop_17_conv2d_73_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_18AssignVariableOp0assignvariableop_18_batch_normalization_41_gammaIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_19AssignVariableOp/assignvariableop_19_batch_normalization_41_betaIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_20AssignVariableOp6assignvariableop_20_batch_normalization_41_moving_meanIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp:assignvariableop_21_batch_normalization_41_moving_varianceIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_22AssignVariableOp#assignvariableop_22_dense_26_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp!assignvariableop_23_dense_26_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_24AssignVariableOp#assignvariableop_24_dense_27_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_25AssignVariableOp!assignvariableop_25_dense_27_biasIdentity_25:output:0"/device:CPU:0*
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
AssignVariableOp_35AssignVariableOp+assignvariableop_35_adam_conv2d_69_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_36AssignVariableOp)assignvariableop_36_adam_conv2d_69_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_37AssignVariableOp+assignvariableop_37_adam_conv2d_70_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_38AssignVariableOp)assignvariableop_38_adam_conv2d_70_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_39AssignVariableOp7assignvariableop_39_adam_batch_normalization_39_gamma_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_40AssignVariableOp6assignvariableop_40_adam_batch_normalization_39_beta_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_41AssignVariableOp+assignvariableop_41_adam_conv2d_71_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_42AssignVariableOp)assignvariableop_42_adam_conv2d_71_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_43AssignVariableOp+assignvariableop_43_adam_conv2d_72_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_44AssignVariableOp)assignvariableop_44_adam_conv2d_72_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_45AssignVariableOp7assignvariableop_45_adam_batch_normalization_40_gamma_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_46AssignVariableOp6assignvariableop_46_adam_batch_normalization_40_beta_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_47AssignVariableOp+assignvariableop_47_adam_conv2d_73_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_48AssignVariableOp)assignvariableop_48_adam_conv2d_73_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_49AssignVariableOp7assignvariableop_49_adam_batch_normalization_41_gamma_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_50AssignVariableOp6assignvariableop_50_adam_batch_normalization_41_beta_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_51AssignVariableOp*assignvariableop_51_adam_dense_26_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_52AssignVariableOp(assignvariableop_52_adam_dense_26_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_53AssignVariableOp*assignvariableop_53_adam_dense_27_kernel_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_54AssignVariableOp(assignvariableop_54_adam_dense_27_bias_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_55AssignVariableOp+assignvariableop_55_adam_conv2d_69_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_56AssignVariableOp)assignvariableop_56_adam_conv2d_69_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_57AssignVariableOp+assignvariableop_57_adam_conv2d_70_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_58AssignVariableOp)assignvariableop_58_adam_conv2d_70_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_59AssignVariableOp7assignvariableop_59_adam_batch_normalization_39_gamma_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_60AssignVariableOp6assignvariableop_60_adam_batch_normalization_39_beta_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_61AssignVariableOp+assignvariableop_61_adam_conv2d_71_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_62AssignVariableOp)assignvariableop_62_adam_conv2d_71_bias_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_63AssignVariableOp+assignvariableop_63_adam_conv2d_72_kernel_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_64AssignVariableOp)assignvariableop_64_adam_conv2d_72_bias_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_65AssignVariableOp7assignvariableop_65_adam_batch_normalization_40_gamma_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_66AssignVariableOp6assignvariableop_66_adam_batch_normalization_40_beta_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_67AssignVariableOp+assignvariableop_67_adam_conv2d_73_kernel_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_68AssignVariableOp)assignvariableop_68_adam_conv2d_73_bias_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_69AssignVariableOp7assignvariableop_69_adam_batch_normalization_41_gamma_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_70AssignVariableOp6assignvariableop_70_adam_batch_normalization_41_beta_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_71AssignVariableOp*assignvariableop_71_adam_dense_26_kernel_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_72AssignVariableOp(assignvariableop_72_adam_dense_26_bias_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_73AssignVariableOp*assignvariableop_73_adam_dense_27_kernel_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_74AssignVariableOp(assignvariableop_74_adam_dense_27_bias_vIdentity_74:output:0"/device:CPU:0*
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
?
?
F__inference_conv2d_72_layer_call_and_return_conditional_losses_2449924

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
?	
?
8__inference_batch_normalization_39_layer_call_fn_2450977

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
S__inference_batch_normalization_39_layer_call_and_return_conditional_losses_2449682?
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
??
?
"__inference__wrapped_model_2449617
conv2d_69_inputP
6sequential_15_conv2d_69_conv2d_readvariableop_resource:@E
7sequential_15_conv2d_69_biasadd_readvariableop_resource:@P
6sequential_15_conv2d_70_conv2d_readvariableop_resource:@@E
7sequential_15_conv2d_70_biasadd_readvariableop_resource:@J
<sequential_15_batch_normalization_39_readvariableop_resource:@L
>sequential_15_batch_normalization_39_readvariableop_1_resource:@[
Msequential_15_batch_normalization_39_fusedbatchnormv3_readvariableop_resource:@]
Osequential_15_batch_normalization_39_fusedbatchnormv3_readvariableop_1_resource:@Q
6sequential_15_conv2d_71_conv2d_readvariableop_resource:@?F
7sequential_15_conv2d_71_biasadd_readvariableop_resource:	?R
6sequential_15_conv2d_72_conv2d_readvariableop_resource:??F
7sequential_15_conv2d_72_biasadd_readvariableop_resource:	?K
<sequential_15_batch_normalization_40_readvariableop_resource:	?M
>sequential_15_batch_normalization_40_readvariableop_1_resource:	?\
Msequential_15_batch_normalization_40_fusedbatchnormv3_readvariableop_resource:	?^
Osequential_15_batch_normalization_40_fusedbatchnormv3_readvariableop_1_resource:	?R
6sequential_15_conv2d_73_conv2d_readvariableop_resource:??F
7sequential_15_conv2d_73_biasadd_readvariableop_resource:	?K
<sequential_15_batch_normalization_41_readvariableop_resource:	?M
>sequential_15_batch_normalization_41_readvariableop_1_resource:	?\
Msequential_15_batch_normalization_41_fusedbatchnormv3_readvariableop_resource:	?^
Osequential_15_batch_normalization_41_fusedbatchnormv3_readvariableop_1_resource:	?I
5sequential_15_dense_26_matmul_readvariableop_resource:
??E
6sequential_15_dense_26_biasadd_readvariableop_resource:	?H
5sequential_15_dense_27_matmul_readvariableop_resource:	?D
6sequential_15_dense_27_biasadd_readvariableop_resource:
identity??Dsequential_15/batch_normalization_39/FusedBatchNormV3/ReadVariableOp?Fsequential_15/batch_normalization_39/FusedBatchNormV3/ReadVariableOp_1?3sequential_15/batch_normalization_39/ReadVariableOp?5sequential_15/batch_normalization_39/ReadVariableOp_1?Dsequential_15/batch_normalization_40/FusedBatchNormV3/ReadVariableOp?Fsequential_15/batch_normalization_40/FusedBatchNormV3/ReadVariableOp_1?3sequential_15/batch_normalization_40/ReadVariableOp?5sequential_15/batch_normalization_40/ReadVariableOp_1?Dsequential_15/batch_normalization_41/FusedBatchNormV3/ReadVariableOp?Fsequential_15/batch_normalization_41/FusedBatchNormV3/ReadVariableOp_1?3sequential_15/batch_normalization_41/ReadVariableOp?5sequential_15/batch_normalization_41/ReadVariableOp_1?.sequential_15/conv2d_69/BiasAdd/ReadVariableOp?-sequential_15/conv2d_69/Conv2D/ReadVariableOp?.sequential_15/conv2d_70/BiasAdd/ReadVariableOp?-sequential_15/conv2d_70/Conv2D/ReadVariableOp?.sequential_15/conv2d_71/BiasAdd/ReadVariableOp?-sequential_15/conv2d_71/Conv2D/ReadVariableOp?.sequential_15/conv2d_72/BiasAdd/ReadVariableOp?-sequential_15/conv2d_72/Conv2D/ReadVariableOp?.sequential_15/conv2d_73/BiasAdd/ReadVariableOp?-sequential_15/conv2d_73/Conv2D/ReadVariableOp?-sequential_15/dense_26/BiasAdd/ReadVariableOp?,sequential_15/dense_26/MatMul/ReadVariableOp?-sequential_15/dense_27/BiasAdd/ReadVariableOp?,sequential_15/dense_27/MatMul/ReadVariableOp?
-sequential_15/conv2d_69/Conv2D/ReadVariableOpReadVariableOp6sequential_15_conv2d_69_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
sequential_15/conv2d_69/Conv2DConv2Dconv2d_69_input5sequential_15/conv2d_69/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????6@*
paddingVALID*
strides
?
.sequential_15/conv2d_69/BiasAdd/ReadVariableOpReadVariableOp7sequential_15_conv2d_69_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
sequential_15/conv2d_69/BiasAddBiasAdd'sequential_15/conv2d_69/Conv2D:output:06sequential_15/conv2d_69/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????6@?
sequential_15/conv2d_69/ReluRelu(sequential_15/conv2d_69/BiasAdd:output:0*
T0*/
_output_shapes
:?????????6@?
-sequential_15/conv2d_70/Conv2D/ReadVariableOpReadVariableOp6sequential_15_conv2d_70_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
sequential_15/conv2d_70/Conv2DConv2D*sequential_15/conv2d_69/Relu:activations:05sequential_15/conv2d_70/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????4@*
paddingVALID*
strides
?
.sequential_15/conv2d_70/BiasAdd/ReadVariableOpReadVariableOp7sequential_15_conv2d_70_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
sequential_15/conv2d_70/BiasAddBiasAdd'sequential_15/conv2d_70/Conv2D:output:06sequential_15/conv2d_70/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????4@?
sequential_15/conv2d_70/ReluRelu(sequential_15/conv2d_70/BiasAdd:output:0*
T0*/
_output_shapes
:?????????4@?
&sequential_15/max_pooling2d_39/MaxPoolMaxPool*sequential_15/conv2d_70/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
?
3sequential_15/batch_normalization_39/ReadVariableOpReadVariableOp<sequential_15_batch_normalization_39_readvariableop_resource*
_output_shapes
:@*
dtype0?
5sequential_15/batch_normalization_39/ReadVariableOp_1ReadVariableOp>sequential_15_batch_normalization_39_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
Dsequential_15/batch_normalization_39/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_15_batch_normalization_39_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
Fsequential_15/batch_normalization_39/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_15_batch_normalization_39_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
5sequential_15/batch_normalization_39/FusedBatchNormV3FusedBatchNormV3/sequential_15/max_pooling2d_39/MaxPool:output:0;sequential_15/batch_normalization_39/ReadVariableOp:value:0=sequential_15/batch_normalization_39/ReadVariableOp_1:value:0Lsequential_15/batch_normalization_39/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_15/batch_normalization_39/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
is_training( ?
-sequential_15/conv2d_71/Conv2D/ReadVariableOpReadVariableOp6sequential_15_conv2d_71_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
sequential_15/conv2d_71/Conv2DConv2D9sequential_15/batch_normalization_39/FusedBatchNormV3:y:05sequential_15/conv2d_71/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????
?*
paddingVALID*
strides
?
.sequential_15/conv2d_71/BiasAdd/ReadVariableOpReadVariableOp7sequential_15_conv2d_71_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
sequential_15/conv2d_71/BiasAddBiasAdd'sequential_15/conv2d_71/Conv2D:output:06sequential_15/conv2d_71/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????
??
sequential_15/conv2d_71/ReluRelu(sequential_15/conv2d_71/BiasAdd:output:0*
T0*0
_output_shapes
:?????????
??
-sequential_15/conv2d_72/Conv2D/ReadVariableOpReadVariableOp6sequential_15_conv2d_72_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
sequential_15/conv2d_72/Conv2DConv2D*sequential_15/conv2d_71/Relu:activations:05sequential_15/conv2d_72/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
?
.sequential_15/conv2d_72/BiasAdd/ReadVariableOpReadVariableOp7sequential_15_conv2d_72_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
sequential_15/conv2d_72/BiasAddBiasAdd'sequential_15/conv2d_72/Conv2D:output:06sequential_15/conv2d_72/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:???????????
sequential_15/conv2d_72/ReluRelu(sequential_15/conv2d_72/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
&sequential_15/max_pooling2d_40/MaxPoolMaxPool*sequential_15/conv2d_72/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
?
3sequential_15/batch_normalization_40/ReadVariableOpReadVariableOp<sequential_15_batch_normalization_40_readvariableop_resource*
_output_shapes	
:?*
dtype0?
5sequential_15/batch_normalization_40/ReadVariableOp_1ReadVariableOp>sequential_15_batch_normalization_40_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
Dsequential_15/batch_normalization_40/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_15_batch_normalization_40_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Fsequential_15/batch_normalization_40/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_15_batch_normalization_40_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
5sequential_15/batch_normalization_40/FusedBatchNormV3FusedBatchNormV3/sequential_15/max_pooling2d_40/MaxPool:output:0;sequential_15/batch_normalization_40/ReadVariableOp:value:0=sequential_15/batch_normalization_40/ReadVariableOp_1:value:0Lsequential_15/batch_normalization_40/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_15/batch_normalization_40/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( ?
-sequential_15/conv2d_73/Conv2D/ReadVariableOpReadVariableOp6sequential_15_conv2d_73_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
sequential_15/conv2d_73/Conv2DConv2D9sequential_15/batch_normalization_40/FusedBatchNormV3:y:05sequential_15/conv2d_73/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	?*
paddingVALID*
strides
?
.sequential_15/conv2d_73/BiasAdd/ReadVariableOpReadVariableOp7sequential_15_conv2d_73_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
sequential_15/conv2d_73/BiasAddBiasAdd'sequential_15/conv2d_73/Conv2D:output:06sequential_15/conv2d_73/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	??
sequential_15/conv2d_73/ReluRelu(sequential_15/conv2d_73/BiasAdd:output:0*
T0*0
_output_shapes
:?????????	??
&sequential_15/max_pooling2d_41/MaxPoolMaxPool*sequential_15/conv2d_73/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
?
3sequential_15/batch_normalization_41/ReadVariableOpReadVariableOp<sequential_15_batch_normalization_41_readvariableop_resource*
_output_shapes	
:?*
dtype0?
5sequential_15/batch_normalization_41/ReadVariableOp_1ReadVariableOp>sequential_15_batch_normalization_41_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
Dsequential_15/batch_normalization_41/FusedBatchNormV3/ReadVariableOpReadVariableOpMsequential_15_batch_normalization_41_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
Fsequential_15/batch_normalization_41/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpOsequential_15_batch_normalization_41_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
5sequential_15/batch_normalization_41/FusedBatchNormV3FusedBatchNormV3/sequential_15/max_pooling2d_41/MaxPool:output:0;sequential_15/batch_normalization_41/ReadVariableOp:value:0=sequential_15/batch_normalization_41/ReadVariableOp_1:value:0Lsequential_15/batch_normalization_41/FusedBatchNormV3/ReadVariableOp:value:0Nsequential_15/batch_normalization_41/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( o
sequential_15/flatten_13/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ?
 sequential_15/flatten_13/ReshapeReshape9sequential_15/batch_normalization_41/FusedBatchNormV3:y:0'sequential_15/flatten_13/Const:output:0*
T0*(
_output_shapes
:???????????
,sequential_15/dense_26/MatMul/ReadVariableOpReadVariableOp5sequential_15_dense_26_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
sequential_15/dense_26/MatMulMatMul)sequential_15/flatten_13/Reshape:output:04sequential_15/dense_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
-sequential_15/dense_26/BiasAdd/ReadVariableOpReadVariableOp6sequential_15_dense_26_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
sequential_15/dense_26/BiasAddBiasAdd'sequential_15/dense_26/MatMul:product:05sequential_15/dense_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????
sequential_15/dense_26/ReluRelu'sequential_15/dense_26/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
,sequential_15/dense_27/MatMul/ReadVariableOpReadVariableOp5sequential_15_dense_27_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
sequential_15/dense_27/MatMulMatMul)sequential_15/dense_26/Relu:activations:04sequential_15/dense_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
-sequential_15/dense_27/BiasAdd/ReadVariableOpReadVariableOp6sequential_15_dense_27_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
sequential_15/dense_27/BiasAddBiasAdd'sequential_15/dense_27/MatMul:product:05sequential_15/dense_27/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
sequential_15/dense_27/SoftmaxSoftmax'sequential_15/dense_27/BiasAdd:output:0*
T0*'
_output_shapes
:?????????w
IdentityIdentity(sequential_15/dense_27/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOpE^sequential_15/batch_normalization_39/FusedBatchNormV3/ReadVariableOpG^sequential_15/batch_normalization_39/FusedBatchNormV3/ReadVariableOp_14^sequential_15/batch_normalization_39/ReadVariableOp6^sequential_15/batch_normalization_39/ReadVariableOp_1E^sequential_15/batch_normalization_40/FusedBatchNormV3/ReadVariableOpG^sequential_15/batch_normalization_40/FusedBatchNormV3/ReadVariableOp_14^sequential_15/batch_normalization_40/ReadVariableOp6^sequential_15/batch_normalization_40/ReadVariableOp_1E^sequential_15/batch_normalization_41/FusedBatchNormV3/ReadVariableOpG^sequential_15/batch_normalization_41/FusedBatchNormV3/ReadVariableOp_14^sequential_15/batch_normalization_41/ReadVariableOp6^sequential_15/batch_normalization_41/ReadVariableOp_1/^sequential_15/conv2d_69/BiasAdd/ReadVariableOp.^sequential_15/conv2d_69/Conv2D/ReadVariableOp/^sequential_15/conv2d_70/BiasAdd/ReadVariableOp.^sequential_15/conv2d_70/Conv2D/ReadVariableOp/^sequential_15/conv2d_71/BiasAdd/ReadVariableOp.^sequential_15/conv2d_71/Conv2D/ReadVariableOp/^sequential_15/conv2d_72/BiasAdd/ReadVariableOp.^sequential_15/conv2d_72/Conv2D/ReadVariableOp/^sequential_15/conv2d_73/BiasAdd/ReadVariableOp.^sequential_15/conv2d_73/Conv2D/ReadVariableOp.^sequential_15/dense_26/BiasAdd/ReadVariableOp-^sequential_15/dense_26/MatMul/ReadVariableOp.^sequential_15/dense_27/BiasAdd/ReadVariableOp-^sequential_15/dense_27/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:?????????8: : : : : : : : : : : : : : : : : : : : : : : : : : 2?
Dsequential_15/batch_normalization_39/FusedBatchNormV3/ReadVariableOpDsequential_15/batch_normalization_39/FusedBatchNormV3/ReadVariableOp2?
Fsequential_15/batch_normalization_39/FusedBatchNormV3/ReadVariableOp_1Fsequential_15/batch_normalization_39/FusedBatchNormV3/ReadVariableOp_12j
3sequential_15/batch_normalization_39/ReadVariableOp3sequential_15/batch_normalization_39/ReadVariableOp2n
5sequential_15/batch_normalization_39/ReadVariableOp_15sequential_15/batch_normalization_39/ReadVariableOp_12?
Dsequential_15/batch_normalization_40/FusedBatchNormV3/ReadVariableOpDsequential_15/batch_normalization_40/FusedBatchNormV3/ReadVariableOp2?
Fsequential_15/batch_normalization_40/FusedBatchNormV3/ReadVariableOp_1Fsequential_15/batch_normalization_40/FusedBatchNormV3/ReadVariableOp_12j
3sequential_15/batch_normalization_40/ReadVariableOp3sequential_15/batch_normalization_40/ReadVariableOp2n
5sequential_15/batch_normalization_40/ReadVariableOp_15sequential_15/batch_normalization_40/ReadVariableOp_12?
Dsequential_15/batch_normalization_41/FusedBatchNormV3/ReadVariableOpDsequential_15/batch_normalization_41/FusedBatchNormV3/ReadVariableOp2?
Fsequential_15/batch_normalization_41/FusedBatchNormV3/ReadVariableOp_1Fsequential_15/batch_normalization_41/FusedBatchNormV3/ReadVariableOp_12j
3sequential_15/batch_normalization_41/ReadVariableOp3sequential_15/batch_normalization_41/ReadVariableOp2n
5sequential_15/batch_normalization_41/ReadVariableOp_15sequential_15/batch_normalization_41/ReadVariableOp_12`
.sequential_15/conv2d_69/BiasAdd/ReadVariableOp.sequential_15/conv2d_69/BiasAdd/ReadVariableOp2^
-sequential_15/conv2d_69/Conv2D/ReadVariableOp-sequential_15/conv2d_69/Conv2D/ReadVariableOp2`
.sequential_15/conv2d_70/BiasAdd/ReadVariableOp.sequential_15/conv2d_70/BiasAdd/ReadVariableOp2^
-sequential_15/conv2d_70/Conv2D/ReadVariableOp-sequential_15/conv2d_70/Conv2D/ReadVariableOp2`
.sequential_15/conv2d_71/BiasAdd/ReadVariableOp.sequential_15/conv2d_71/BiasAdd/ReadVariableOp2^
-sequential_15/conv2d_71/Conv2D/ReadVariableOp-sequential_15/conv2d_71/Conv2D/ReadVariableOp2`
.sequential_15/conv2d_72/BiasAdd/ReadVariableOp.sequential_15/conv2d_72/BiasAdd/ReadVariableOp2^
-sequential_15/conv2d_72/Conv2D/ReadVariableOp-sequential_15/conv2d_72/Conv2D/ReadVariableOp2`
.sequential_15/conv2d_73/BiasAdd/ReadVariableOp.sequential_15/conv2d_73/BiasAdd/ReadVariableOp2^
-sequential_15/conv2d_73/Conv2D/ReadVariableOp-sequential_15/conv2d_73/Conv2D/ReadVariableOp2^
-sequential_15/dense_26/BiasAdd/ReadVariableOp-sequential_15/dense_26/BiasAdd/ReadVariableOp2\
,sequential_15/dense_26/MatMul/ReadVariableOp,sequential_15/dense_26/MatMul/ReadVariableOp2^
-sequential_15/dense_27/BiasAdd/ReadVariableOp-sequential_15/dense_27/BiasAdd/ReadVariableOp2\
,sequential_15/dense_27/MatMul/ReadVariableOp,sequential_15/dense_27/MatMul/ReadVariableOp:` \
/
_output_shapes
:?????????8
)
_user_specified_nameconv2d_69_input
?
i
M__inference_max_pooling2d_40_layer_call_and_return_conditional_losses_2451063

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
8__inference_batch_normalization_41_layer_call_fn_2451181

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
S__inference_batch_normalization_41_layer_call_and_return_conditional_losses_2449834?
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
?I
?
J__inference_sequential_15_layer_call_and_return_conditional_losses_2450452
conv2d_69_input+
conv2d_69_2450385:@
conv2d_69_2450387:@+
conv2d_70_2450390:@@
conv2d_70_2450392:@,
batch_normalization_39_2450396:@,
batch_normalization_39_2450398:@,
batch_normalization_39_2450400:@,
batch_normalization_39_2450402:@,
conv2d_71_2450405:@? 
conv2d_71_2450407:	?-
conv2d_72_2450410:?? 
conv2d_72_2450412:	?-
batch_normalization_40_2450416:	?-
batch_normalization_40_2450418:	?-
batch_normalization_40_2450420:	?-
batch_normalization_40_2450422:	?-
conv2d_73_2450425:?? 
conv2d_73_2450427:	?-
batch_normalization_41_2450431:	?-
batch_normalization_41_2450433:	?-
batch_normalization_41_2450435:	?-
batch_normalization_41_2450437:	?$
dense_26_2450441:
??
dense_26_2450443:	?#
dense_27_2450446:	?
dense_27_2450448:
identity??.batch_normalization_39/StatefulPartitionedCall?.batch_normalization_40/StatefulPartitionedCall?.batch_normalization_41/StatefulPartitionedCall?!conv2d_69/StatefulPartitionedCall?!conv2d_70/StatefulPartitionedCall?!conv2d_71/StatefulPartitionedCall?!conv2d_72/StatefulPartitionedCall?!conv2d_73/StatefulPartitionedCall? dense_26/StatefulPartitionedCall? dense_27/StatefulPartitionedCall?
!conv2d_69/StatefulPartitionedCallStatefulPartitionedCallconv2d_69_inputconv2d_69_2450385conv2d_69_2450387*
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
F__inference_conv2d_69_layer_call_and_return_conditional_losses_2449863?
!conv2d_70/StatefulPartitionedCallStatefulPartitionedCall*conv2d_69/StatefulPartitionedCall:output:0conv2d_70_2450390conv2d_70_2450392*
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
F__inference_conv2d_70_layer_call_and_return_conditional_losses_2449880?
 max_pooling2d_39/PartitionedCallPartitionedCall*conv2d_70/StatefulPartitionedCall:output:0*
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
M__inference_max_pooling2d_39_layer_call_and_return_conditional_losses_2449626?
.batch_normalization_39/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_39/PartitionedCall:output:0batch_normalization_39_2450396batch_normalization_39_2450398batch_normalization_39_2450400batch_normalization_39_2450402*
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
S__inference_batch_normalization_39_layer_call_and_return_conditional_losses_2449651?
!conv2d_71/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_39/StatefulPartitionedCall:output:0conv2d_71_2450405conv2d_71_2450407*
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
F__inference_conv2d_71_layer_call_and_return_conditional_losses_2449907?
!conv2d_72/StatefulPartitionedCallStatefulPartitionedCall*conv2d_71/StatefulPartitionedCall:output:0conv2d_72_2450410conv2d_72_2450412*
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
F__inference_conv2d_72_layer_call_and_return_conditional_losses_2449924?
 max_pooling2d_40/PartitionedCallPartitionedCall*conv2d_72/StatefulPartitionedCall:output:0*
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
M__inference_max_pooling2d_40_layer_call_and_return_conditional_losses_2449702?
.batch_normalization_40/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_40/PartitionedCall:output:0batch_normalization_40_2450416batch_normalization_40_2450418batch_normalization_40_2450420batch_normalization_40_2450422*
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
S__inference_batch_normalization_40_layer_call_and_return_conditional_losses_2449727?
!conv2d_73/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_40/StatefulPartitionedCall:output:0conv2d_73_2450425conv2d_73_2450427*
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
F__inference_conv2d_73_layer_call_and_return_conditional_losses_2449951?
 max_pooling2d_41/PartitionedCallPartitionedCall*conv2d_73/StatefulPartitionedCall:output:0*
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
M__inference_max_pooling2d_41_layer_call_and_return_conditional_losses_2449778?
.batch_normalization_41/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_41/PartitionedCall:output:0batch_normalization_41_2450431batch_normalization_41_2450433batch_normalization_41_2450435batch_normalization_41_2450437*
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
S__inference_batch_normalization_41_layer_call_and_return_conditional_losses_2449803?
flatten_13/PartitionedCallPartitionedCall7batch_normalization_41/StatefulPartitionedCall:output:0*
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
G__inference_flatten_13_layer_call_and_return_conditional_losses_2449973?
 dense_26/StatefulPartitionedCallStatefulPartitionedCall#flatten_13/PartitionedCall:output:0dense_26_2450441dense_26_2450443*
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
E__inference_dense_26_layer_call_and_return_conditional_losses_2449986?
 dense_27/StatefulPartitionedCallStatefulPartitionedCall)dense_26/StatefulPartitionedCall:output:0dense_27_2450446dense_27_2450448*
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
E__inference_dense_27_layer_call_and_return_conditional_losses_2450003x
IdentityIdentity)dense_27/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp/^batch_normalization_39/StatefulPartitionedCall/^batch_normalization_40/StatefulPartitionedCall/^batch_normalization_41/StatefulPartitionedCall"^conv2d_69/StatefulPartitionedCall"^conv2d_70/StatefulPartitionedCall"^conv2d_71/StatefulPartitionedCall"^conv2d_72/StatefulPartitionedCall"^conv2d_73/StatefulPartitionedCall!^dense_26/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:?????????8: : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_39/StatefulPartitionedCall.batch_normalization_39/StatefulPartitionedCall2`
.batch_normalization_40/StatefulPartitionedCall.batch_normalization_40/StatefulPartitionedCall2`
.batch_normalization_41/StatefulPartitionedCall.batch_normalization_41/StatefulPartitionedCall2F
!conv2d_69/StatefulPartitionedCall!conv2d_69/StatefulPartitionedCall2F
!conv2d_70/StatefulPartitionedCall!conv2d_70/StatefulPartitionedCall2F
!conv2d_71/StatefulPartitionedCall!conv2d_71/StatefulPartitionedCall2F
!conv2d_72/StatefulPartitionedCall!conv2d_72/StatefulPartitionedCall2F
!conv2d_73/StatefulPartitionedCall!conv2d_73/StatefulPartitionedCall2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall:` \
/
_output_shapes
:?????????8
)
_user_specified_nameconv2d_69_input
?
N
2__inference_max_pooling2d_41_layer_call_fn_2451150

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
M__inference_max_pooling2d_41_layer_call_and_return_conditional_losses_2449778?
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
?
?
S__inference_batch_normalization_40_layer_call_and_return_conditional_losses_2449758

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
E__inference_dense_27_layer_call_and_return_conditional_losses_2451268

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
?
i
M__inference_max_pooling2d_40_layer_call_and_return_conditional_losses_2449702

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
?
?
+__inference_conv2d_70_layer_call_fn_2450930

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
F__inference_conv2d_70_layer_call_and_return_conditional_losses_2449880w
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
?
?
F__inference_conv2d_73_layer_call_and_return_conditional_losses_2449951

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
?
N
2__inference_max_pooling2d_39_layer_call_fn_2450946

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
M__inference_max_pooling2d_39_layer_call_and_return_conditional_losses_2449626?
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
8__inference_batch_normalization_39_layer_call_fn_2450964

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
S__inference_batch_normalization_39_layer_call_and_return_conditional_losses_2449651?
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
?	
?
8__inference_batch_normalization_40_layer_call_fn_2451076

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
S__inference_batch_normalization_40_layer_call_and_return_conditional_losses_2449727?
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
?H
?
J__inference_sequential_15_layer_call_and_return_conditional_losses_2450010

inputs+
conv2d_69_2449864:@
conv2d_69_2449866:@+
conv2d_70_2449881:@@
conv2d_70_2449883:@,
batch_normalization_39_2449887:@,
batch_normalization_39_2449889:@,
batch_normalization_39_2449891:@,
batch_normalization_39_2449893:@,
conv2d_71_2449908:@? 
conv2d_71_2449910:	?-
conv2d_72_2449925:?? 
conv2d_72_2449927:	?-
batch_normalization_40_2449931:	?-
batch_normalization_40_2449933:	?-
batch_normalization_40_2449935:	?-
batch_normalization_40_2449937:	?-
conv2d_73_2449952:?? 
conv2d_73_2449954:	?-
batch_normalization_41_2449958:	?-
batch_normalization_41_2449960:	?-
batch_normalization_41_2449962:	?-
batch_normalization_41_2449964:	?$
dense_26_2449987:
??
dense_26_2449989:	?#
dense_27_2450004:	?
dense_27_2450006:
identity??.batch_normalization_39/StatefulPartitionedCall?.batch_normalization_40/StatefulPartitionedCall?.batch_normalization_41/StatefulPartitionedCall?!conv2d_69/StatefulPartitionedCall?!conv2d_70/StatefulPartitionedCall?!conv2d_71/StatefulPartitionedCall?!conv2d_72/StatefulPartitionedCall?!conv2d_73/StatefulPartitionedCall? dense_26/StatefulPartitionedCall? dense_27/StatefulPartitionedCall?
!conv2d_69/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_69_2449864conv2d_69_2449866*
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
F__inference_conv2d_69_layer_call_and_return_conditional_losses_2449863?
!conv2d_70/StatefulPartitionedCallStatefulPartitionedCall*conv2d_69/StatefulPartitionedCall:output:0conv2d_70_2449881conv2d_70_2449883*
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
F__inference_conv2d_70_layer_call_and_return_conditional_losses_2449880?
 max_pooling2d_39/PartitionedCallPartitionedCall*conv2d_70/StatefulPartitionedCall:output:0*
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
M__inference_max_pooling2d_39_layer_call_and_return_conditional_losses_2449626?
.batch_normalization_39/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_39/PartitionedCall:output:0batch_normalization_39_2449887batch_normalization_39_2449889batch_normalization_39_2449891batch_normalization_39_2449893*
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
S__inference_batch_normalization_39_layer_call_and_return_conditional_losses_2449651?
!conv2d_71/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_39/StatefulPartitionedCall:output:0conv2d_71_2449908conv2d_71_2449910*
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
F__inference_conv2d_71_layer_call_and_return_conditional_losses_2449907?
!conv2d_72/StatefulPartitionedCallStatefulPartitionedCall*conv2d_71/StatefulPartitionedCall:output:0conv2d_72_2449925conv2d_72_2449927*
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
F__inference_conv2d_72_layer_call_and_return_conditional_losses_2449924?
 max_pooling2d_40/PartitionedCallPartitionedCall*conv2d_72/StatefulPartitionedCall:output:0*
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
M__inference_max_pooling2d_40_layer_call_and_return_conditional_losses_2449702?
.batch_normalization_40/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_40/PartitionedCall:output:0batch_normalization_40_2449931batch_normalization_40_2449933batch_normalization_40_2449935batch_normalization_40_2449937*
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
S__inference_batch_normalization_40_layer_call_and_return_conditional_losses_2449727?
!conv2d_73/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_40/StatefulPartitionedCall:output:0conv2d_73_2449952conv2d_73_2449954*
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
F__inference_conv2d_73_layer_call_and_return_conditional_losses_2449951?
 max_pooling2d_41/PartitionedCallPartitionedCall*conv2d_73/StatefulPartitionedCall:output:0*
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
M__inference_max_pooling2d_41_layer_call_and_return_conditional_losses_2449778?
.batch_normalization_41/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_41/PartitionedCall:output:0batch_normalization_41_2449958batch_normalization_41_2449960batch_normalization_41_2449962batch_normalization_41_2449964*
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
S__inference_batch_normalization_41_layer_call_and_return_conditional_losses_2449803?
flatten_13/PartitionedCallPartitionedCall7batch_normalization_41/StatefulPartitionedCall:output:0*
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
G__inference_flatten_13_layer_call_and_return_conditional_losses_2449973?
 dense_26/StatefulPartitionedCallStatefulPartitionedCall#flatten_13/PartitionedCall:output:0dense_26_2449987dense_26_2449989*
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
E__inference_dense_26_layer_call_and_return_conditional_losses_2449986?
 dense_27/StatefulPartitionedCallStatefulPartitionedCall)dense_26/StatefulPartitionedCall:output:0dense_27_2450004dense_27_2450006*
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
E__inference_dense_27_layer_call_and_return_conditional_losses_2450003x
IdentityIdentity)dense_27/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp/^batch_normalization_39/StatefulPartitionedCall/^batch_normalization_40/StatefulPartitionedCall/^batch_normalization_41/StatefulPartitionedCall"^conv2d_69/StatefulPartitionedCall"^conv2d_70/StatefulPartitionedCall"^conv2d_71/StatefulPartitionedCall"^conv2d_72/StatefulPartitionedCall"^conv2d_73/StatefulPartitionedCall!^dense_26/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:?????????8: : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_39/StatefulPartitionedCall.batch_normalization_39/StatefulPartitionedCall2`
.batch_normalization_40/StatefulPartitionedCall.batch_normalization_40/StatefulPartitionedCall2`
.batch_normalization_41/StatefulPartitionedCall.batch_normalization_41/StatefulPartitionedCall2F
!conv2d_69/StatefulPartitionedCall!conv2d_69/StatefulPartitionedCall2F
!conv2d_70/StatefulPartitionedCall!conv2d_70/StatefulPartitionedCall2F
!conv2d_71/StatefulPartitionedCall!conv2d_71/StatefulPartitionedCall2F
!conv2d_72/StatefulPartitionedCall!conv2d_72/StatefulPartitionedCall2F
!conv2d_73/StatefulPartitionedCall!conv2d_73/StatefulPartitionedCall2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall:W S
/
_output_shapes
:?????????8
 
_user_specified_nameinputs
?
?
+__inference_conv2d_73_layer_call_fn_2451134

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
F__inference_conv2d_73_layer_call_and_return_conditional_losses_2449951x
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
%__inference_signature_wrapper_2450587
conv2d_69_input!
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
StatefulPartitionedCallStatefulPartitionedCallconv2d_69_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
"__inference__wrapped_model_2449617o
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
_user_specified_nameconv2d_69_input
?
?
S__inference_batch_normalization_39_layer_call_and_return_conditional_losses_2449651

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
?
i
M__inference_max_pooling2d_41_layer_call_and_return_conditional_losses_2449778

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
?
F__inference_conv2d_72_layer_call_and_return_conditional_losses_2451053

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
?
i
M__inference_max_pooling2d_41_layer_call_and_return_conditional_losses_2451155

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
?
c
G__inference_flatten_13_layer_call_and_return_conditional_losses_2449973

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
?!
 __inference__traced_save_2451516
file_prefix/
+savev2_conv2d_69_kernel_read_readvariableop-
)savev2_conv2d_69_bias_read_readvariableop/
+savev2_conv2d_70_kernel_read_readvariableop-
)savev2_conv2d_70_bias_read_readvariableop;
7savev2_batch_normalization_39_gamma_read_readvariableop:
6savev2_batch_normalization_39_beta_read_readvariableopA
=savev2_batch_normalization_39_moving_mean_read_readvariableopE
Asavev2_batch_normalization_39_moving_variance_read_readvariableop/
+savev2_conv2d_71_kernel_read_readvariableop-
)savev2_conv2d_71_bias_read_readvariableop/
+savev2_conv2d_72_kernel_read_readvariableop-
)savev2_conv2d_72_bias_read_readvariableop;
7savev2_batch_normalization_40_gamma_read_readvariableop:
6savev2_batch_normalization_40_beta_read_readvariableopA
=savev2_batch_normalization_40_moving_mean_read_readvariableopE
Asavev2_batch_normalization_40_moving_variance_read_readvariableop/
+savev2_conv2d_73_kernel_read_readvariableop-
)savev2_conv2d_73_bias_read_readvariableop;
7savev2_batch_normalization_41_gamma_read_readvariableop:
6savev2_batch_normalization_41_beta_read_readvariableopA
=savev2_batch_normalization_41_moving_mean_read_readvariableopE
Asavev2_batch_normalization_41_moving_variance_read_readvariableop.
*savev2_dense_26_kernel_read_readvariableop,
(savev2_dense_26_bias_read_readvariableop.
*savev2_dense_27_kernel_read_readvariableop,
(savev2_dense_27_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop6
2savev2_adam_conv2d_69_kernel_m_read_readvariableop4
0savev2_adam_conv2d_69_bias_m_read_readvariableop6
2savev2_adam_conv2d_70_kernel_m_read_readvariableop4
0savev2_adam_conv2d_70_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_39_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_39_beta_m_read_readvariableop6
2savev2_adam_conv2d_71_kernel_m_read_readvariableop4
0savev2_adam_conv2d_71_bias_m_read_readvariableop6
2savev2_adam_conv2d_72_kernel_m_read_readvariableop4
0savev2_adam_conv2d_72_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_40_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_40_beta_m_read_readvariableop6
2savev2_adam_conv2d_73_kernel_m_read_readvariableop4
0savev2_adam_conv2d_73_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_41_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_41_beta_m_read_readvariableop5
1savev2_adam_dense_26_kernel_m_read_readvariableop3
/savev2_adam_dense_26_bias_m_read_readvariableop5
1savev2_adam_dense_27_kernel_m_read_readvariableop3
/savev2_adam_dense_27_bias_m_read_readvariableop6
2savev2_adam_conv2d_69_kernel_v_read_readvariableop4
0savev2_adam_conv2d_69_bias_v_read_readvariableop6
2savev2_adam_conv2d_70_kernel_v_read_readvariableop4
0savev2_adam_conv2d_70_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_39_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_39_beta_v_read_readvariableop6
2savev2_adam_conv2d_71_kernel_v_read_readvariableop4
0savev2_adam_conv2d_71_bias_v_read_readvariableop6
2savev2_adam_conv2d_72_kernel_v_read_readvariableop4
0savev2_adam_conv2d_72_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_40_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_40_beta_v_read_readvariableop6
2savev2_adam_conv2d_73_kernel_v_read_readvariableop4
0savev2_adam_conv2d_73_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_41_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_41_beta_v_read_readvariableop5
1savev2_adam_dense_26_kernel_v_read_readvariableop3
/savev2_adam_dense_26_bias_v_read_readvariableop5
1savev2_adam_dense_27_kernel_v_read_readvariableop3
/savev2_adam_dense_27_bias_v_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_69_kernel_read_readvariableop)savev2_conv2d_69_bias_read_readvariableop+savev2_conv2d_70_kernel_read_readvariableop)savev2_conv2d_70_bias_read_readvariableop7savev2_batch_normalization_39_gamma_read_readvariableop6savev2_batch_normalization_39_beta_read_readvariableop=savev2_batch_normalization_39_moving_mean_read_readvariableopAsavev2_batch_normalization_39_moving_variance_read_readvariableop+savev2_conv2d_71_kernel_read_readvariableop)savev2_conv2d_71_bias_read_readvariableop+savev2_conv2d_72_kernel_read_readvariableop)savev2_conv2d_72_bias_read_readvariableop7savev2_batch_normalization_40_gamma_read_readvariableop6savev2_batch_normalization_40_beta_read_readvariableop=savev2_batch_normalization_40_moving_mean_read_readvariableopAsavev2_batch_normalization_40_moving_variance_read_readvariableop+savev2_conv2d_73_kernel_read_readvariableop)savev2_conv2d_73_bias_read_readvariableop7savev2_batch_normalization_41_gamma_read_readvariableop6savev2_batch_normalization_41_beta_read_readvariableop=savev2_batch_normalization_41_moving_mean_read_readvariableopAsavev2_batch_normalization_41_moving_variance_read_readvariableop*savev2_dense_26_kernel_read_readvariableop(savev2_dense_26_bias_read_readvariableop*savev2_dense_27_kernel_read_readvariableop(savev2_dense_27_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop2savev2_adam_conv2d_69_kernel_m_read_readvariableop0savev2_adam_conv2d_69_bias_m_read_readvariableop2savev2_adam_conv2d_70_kernel_m_read_readvariableop0savev2_adam_conv2d_70_bias_m_read_readvariableop>savev2_adam_batch_normalization_39_gamma_m_read_readvariableop=savev2_adam_batch_normalization_39_beta_m_read_readvariableop2savev2_adam_conv2d_71_kernel_m_read_readvariableop0savev2_adam_conv2d_71_bias_m_read_readvariableop2savev2_adam_conv2d_72_kernel_m_read_readvariableop0savev2_adam_conv2d_72_bias_m_read_readvariableop>savev2_adam_batch_normalization_40_gamma_m_read_readvariableop=savev2_adam_batch_normalization_40_beta_m_read_readvariableop2savev2_adam_conv2d_73_kernel_m_read_readvariableop0savev2_adam_conv2d_73_bias_m_read_readvariableop>savev2_adam_batch_normalization_41_gamma_m_read_readvariableop=savev2_adam_batch_normalization_41_beta_m_read_readvariableop1savev2_adam_dense_26_kernel_m_read_readvariableop/savev2_adam_dense_26_bias_m_read_readvariableop1savev2_adam_dense_27_kernel_m_read_readvariableop/savev2_adam_dense_27_bias_m_read_readvariableop2savev2_adam_conv2d_69_kernel_v_read_readvariableop0savev2_adam_conv2d_69_bias_v_read_readvariableop2savev2_adam_conv2d_70_kernel_v_read_readvariableop0savev2_adam_conv2d_70_bias_v_read_readvariableop>savev2_adam_batch_normalization_39_gamma_v_read_readvariableop=savev2_adam_batch_normalization_39_beta_v_read_readvariableop2savev2_adam_conv2d_71_kernel_v_read_readvariableop0savev2_adam_conv2d_71_bias_v_read_readvariableop2savev2_adam_conv2d_72_kernel_v_read_readvariableop0savev2_adam_conv2d_72_bias_v_read_readvariableop>savev2_adam_batch_normalization_40_gamma_v_read_readvariableop=savev2_adam_batch_normalization_40_beta_v_read_readvariableop2savev2_adam_conv2d_73_kernel_v_read_readvariableop0savev2_adam_conv2d_73_bias_v_read_readvariableop>savev2_adam_batch_normalization_41_gamma_v_read_readvariableop=savev2_adam_batch_normalization_41_beta_v_read_readvariableop1savev2_adam_dense_26_kernel_v_read_readvariableop/savev2_adam_dense_26_bias_v_read_readvariableop1savev2_adam_dense_27_kernel_v_read_readvariableop/savev2_adam_dense_27_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
?
N
2__inference_max_pooling2d_40_layer_call_fn_2451058

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
M__inference_max_pooling2d_40_layer_call_and_return_conditional_losses_2449702?
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
?
?
*__inference_dense_26_layer_call_fn_2451237

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
E__inference_dense_26_layer_call_and_return_conditional_losses_2449986p
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

?
E__inference_dense_26_layer_call_and_return_conditional_losses_2449986

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
?
i
M__inference_max_pooling2d_39_layer_call_and_return_conditional_losses_2450951

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
F__inference_conv2d_70_layer_call_and_return_conditional_losses_2449880

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
F__inference_conv2d_69_layer_call_and_return_conditional_losses_2449863

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
/__inference_sequential_15_layer_call_fn_2450644

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
J__inference_sequential_15_layer_call_and_return_conditional_losses_2450010o
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
?
?
/__inference_sequential_15_layer_call_fn_2450701

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
J__inference_sequential_15_layer_call_and_return_conditional_losses_2450270o
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
?
?
S__inference_batch_normalization_41_layer_call_and_return_conditional_losses_2449803

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
F__inference_conv2d_70_layer_call_and_return_conditional_losses_2450941

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
?
c
G__inference_flatten_13_layer_call_and_return_conditional_losses_2451228

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
?H
?
J__inference_sequential_15_layer_call_and_return_conditional_losses_2450270

inputs+
conv2d_69_2450203:@
conv2d_69_2450205:@+
conv2d_70_2450208:@@
conv2d_70_2450210:@,
batch_normalization_39_2450214:@,
batch_normalization_39_2450216:@,
batch_normalization_39_2450218:@,
batch_normalization_39_2450220:@,
conv2d_71_2450223:@? 
conv2d_71_2450225:	?-
conv2d_72_2450228:?? 
conv2d_72_2450230:	?-
batch_normalization_40_2450234:	?-
batch_normalization_40_2450236:	?-
batch_normalization_40_2450238:	?-
batch_normalization_40_2450240:	?-
conv2d_73_2450243:?? 
conv2d_73_2450245:	?-
batch_normalization_41_2450249:	?-
batch_normalization_41_2450251:	?-
batch_normalization_41_2450253:	?-
batch_normalization_41_2450255:	?$
dense_26_2450259:
??
dense_26_2450261:	?#
dense_27_2450264:	?
dense_27_2450266:
identity??.batch_normalization_39/StatefulPartitionedCall?.batch_normalization_40/StatefulPartitionedCall?.batch_normalization_41/StatefulPartitionedCall?!conv2d_69/StatefulPartitionedCall?!conv2d_70/StatefulPartitionedCall?!conv2d_71/StatefulPartitionedCall?!conv2d_72/StatefulPartitionedCall?!conv2d_73/StatefulPartitionedCall? dense_26/StatefulPartitionedCall? dense_27/StatefulPartitionedCall?
!conv2d_69/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_69_2450203conv2d_69_2450205*
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
F__inference_conv2d_69_layer_call_and_return_conditional_losses_2449863?
!conv2d_70/StatefulPartitionedCallStatefulPartitionedCall*conv2d_69/StatefulPartitionedCall:output:0conv2d_70_2450208conv2d_70_2450210*
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
F__inference_conv2d_70_layer_call_and_return_conditional_losses_2449880?
 max_pooling2d_39/PartitionedCallPartitionedCall*conv2d_70/StatefulPartitionedCall:output:0*
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
M__inference_max_pooling2d_39_layer_call_and_return_conditional_losses_2449626?
.batch_normalization_39/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_39/PartitionedCall:output:0batch_normalization_39_2450214batch_normalization_39_2450216batch_normalization_39_2450218batch_normalization_39_2450220*
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
S__inference_batch_normalization_39_layer_call_and_return_conditional_losses_2449682?
!conv2d_71/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_39/StatefulPartitionedCall:output:0conv2d_71_2450223conv2d_71_2450225*
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
F__inference_conv2d_71_layer_call_and_return_conditional_losses_2449907?
!conv2d_72/StatefulPartitionedCallStatefulPartitionedCall*conv2d_71/StatefulPartitionedCall:output:0conv2d_72_2450228conv2d_72_2450230*
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
F__inference_conv2d_72_layer_call_and_return_conditional_losses_2449924?
 max_pooling2d_40/PartitionedCallPartitionedCall*conv2d_72/StatefulPartitionedCall:output:0*
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
M__inference_max_pooling2d_40_layer_call_and_return_conditional_losses_2449702?
.batch_normalization_40/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_40/PartitionedCall:output:0batch_normalization_40_2450234batch_normalization_40_2450236batch_normalization_40_2450238batch_normalization_40_2450240*
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
S__inference_batch_normalization_40_layer_call_and_return_conditional_losses_2449758?
!conv2d_73/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_40/StatefulPartitionedCall:output:0conv2d_73_2450243conv2d_73_2450245*
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
F__inference_conv2d_73_layer_call_and_return_conditional_losses_2449951?
 max_pooling2d_41/PartitionedCallPartitionedCall*conv2d_73/StatefulPartitionedCall:output:0*
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
M__inference_max_pooling2d_41_layer_call_and_return_conditional_losses_2449778?
.batch_normalization_41/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_41/PartitionedCall:output:0batch_normalization_41_2450249batch_normalization_41_2450251batch_normalization_41_2450253batch_normalization_41_2450255*
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
S__inference_batch_normalization_41_layer_call_and_return_conditional_losses_2449834?
flatten_13/PartitionedCallPartitionedCall7batch_normalization_41/StatefulPartitionedCall:output:0*
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
G__inference_flatten_13_layer_call_and_return_conditional_losses_2449973?
 dense_26/StatefulPartitionedCallStatefulPartitionedCall#flatten_13/PartitionedCall:output:0dense_26_2450259dense_26_2450261*
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
E__inference_dense_26_layer_call_and_return_conditional_losses_2449986?
 dense_27/StatefulPartitionedCallStatefulPartitionedCall)dense_26/StatefulPartitionedCall:output:0dense_27_2450264dense_27_2450266*
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
E__inference_dense_27_layer_call_and_return_conditional_losses_2450003x
IdentityIdentity)dense_27/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp/^batch_normalization_39/StatefulPartitionedCall/^batch_normalization_40/StatefulPartitionedCall/^batch_normalization_41/StatefulPartitionedCall"^conv2d_69/StatefulPartitionedCall"^conv2d_70/StatefulPartitionedCall"^conv2d_71/StatefulPartitionedCall"^conv2d_72/StatefulPartitionedCall"^conv2d_73/StatefulPartitionedCall!^dense_26/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:?????????8: : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_39/StatefulPartitionedCall.batch_normalization_39/StatefulPartitionedCall2`
.batch_normalization_40/StatefulPartitionedCall.batch_normalization_40/StatefulPartitionedCall2`
.batch_normalization_41/StatefulPartitionedCall.batch_normalization_41/StatefulPartitionedCall2F
!conv2d_69/StatefulPartitionedCall!conv2d_69/StatefulPartitionedCall2F
!conv2d_70/StatefulPartitionedCall!conv2d_70/StatefulPartitionedCall2F
!conv2d_71/StatefulPartitionedCall!conv2d_71/StatefulPartitionedCall2F
!conv2d_72/StatefulPartitionedCall!conv2d_72/StatefulPartitionedCall2F
!conv2d_73/StatefulPartitionedCall!conv2d_73/StatefulPartitionedCall2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall:W S
/
_output_shapes
:?????????8
 
_user_specified_nameinputs
?
?
S__inference_batch_normalization_39_layer_call_and_return_conditional_losses_2450995

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
?

?
E__inference_dense_26_layer_call_and_return_conditional_losses_2451248

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
?I
?
J__inference_sequential_15_layer_call_and_return_conditional_losses_2450522
conv2d_69_input+
conv2d_69_2450455:@
conv2d_69_2450457:@+
conv2d_70_2450460:@@
conv2d_70_2450462:@,
batch_normalization_39_2450466:@,
batch_normalization_39_2450468:@,
batch_normalization_39_2450470:@,
batch_normalization_39_2450472:@,
conv2d_71_2450475:@? 
conv2d_71_2450477:	?-
conv2d_72_2450480:?? 
conv2d_72_2450482:	?-
batch_normalization_40_2450486:	?-
batch_normalization_40_2450488:	?-
batch_normalization_40_2450490:	?-
batch_normalization_40_2450492:	?-
conv2d_73_2450495:?? 
conv2d_73_2450497:	?-
batch_normalization_41_2450501:	?-
batch_normalization_41_2450503:	?-
batch_normalization_41_2450505:	?-
batch_normalization_41_2450507:	?$
dense_26_2450511:
??
dense_26_2450513:	?#
dense_27_2450516:	?
dense_27_2450518:
identity??.batch_normalization_39/StatefulPartitionedCall?.batch_normalization_40/StatefulPartitionedCall?.batch_normalization_41/StatefulPartitionedCall?!conv2d_69/StatefulPartitionedCall?!conv2d_70/StatefulPartitionedCall?!conv2d_71/StatefulPartitionedCall?!conv2d_72/StatefulPartitionedCall?!conv2d_73/StatefulPartitionedCall? dense_26/StatefulPartitionedCall? dense_27/StatefulPartitionedCall?
!conv2d_69/StatefulPartitionedCallStatefulPartitionedCallconv2d_69_inputconv2d_69_2450455conv2d_69_2450457*
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
F__inference_conv2d_69_layer_call_and_return_conditional_losses_2449863?
!conv2d_70/StatefulPartitionedCallStatefulPartitionedCall*conv2d_69/StatefulPartitionedCall:output:0conv2d_70_2450460conv2d_70_2450462*
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
F__inference_conv2d_70_layer_call_and_return_conditional_losses_2449880?
 max_pooling2d_39/PartitionedCallPartitionedCall*conv2d_70/StatefulPartitionedCall:output:0*
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
M__inference_max_pooling2d_39_layer_call_and_return_conditional_losses_2449626?
.batch_normalization_39/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_39/PartitionedCall:output:0batch_normalization_39_2450466batch_normalization_39_2450468batch_normalization_39_2450470batch_normalization_39_2450472*
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
S__inference_batch_normalization_39_layer_call_and_return_conditional_losses_2449682?
!conv2d_71/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_39/StatefulPartitionedCall:output:0conv2d_71_2450475conv2d_71_2450477*
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
F__inference_conv2d_71_layer_call_and_return_conditional_losses_2449907?
!conv2d_72/StatefulPartitionedCallStatefulPartitionedCall*conv2d_71/StatefulPartitionedCall:output:0conv2d_72_2450480conv2d_72_2450482*
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
F__inference_conv2d_72_layer_call_and_return_conditional_losses_2449924?
 max_pooling2d_40/PartitionedCallPartitionedCall*conv2d_72/StatefulPartitionedCall:output:0*
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
M__inference_max_pooling2d_40_layer_call_and_return_conditional_losses_2449702?
.batch_normalization_40/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_40/PartitionedCall:output:0batch_normalization_40_2450486batch_normalization_40_2450488batch_normalization_40_2450490batch_normalization_40_2450492*
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
S__inference_batch_normalization_40_layer_call_and_return_conditional_losses_2449758?
!conv2d_73/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_40/StatefulPartitionedCall:output:0conv2d_73_2450495conv2d_73_2450497*
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
F__inference_conv2d_73_layer_call_and_return_conditional_losses_2449951?
 max_pooling2d_41/PartitionedCallPartitionedCall*conv2d_73/StatefulPartitionedCall:output:0*
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
M__inference_max_pooling2d_41_layer_call_and_return_conditional_losses_2449778?
.batch_normalization_41/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_41/PartitionedCall:output:0batch_normalization_41_2450501batch_normalization_41_2450503batch_normalization_41_2450505batch_normalization_41_2450507*
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
S__inference_batch_normalization_41_layer_call_and_return_conditional_losses_2449834?
flatten_13/PartitionedCallPartitionedCall7batch_normalization_41/StatefulPartitionedCall:output:0*
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
G__inference_flatten_13_layer_call_and_return_conditional_losses_2449973?
 dense_26/StatefulPartitionedCallStatefulPartitionedCall#flatten_13/PartitionedCall:output:0dense_26_2450511dense_26_2450513*
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
E__inference_dense_26_layer_call_and_return_conditional_losses_2449986?
 dense_27/StatefulPartitionedCallStatefulPartitionedCall)dense_26/StatefulPartitionedCall:output:0dense_27_2450516dense_27_2450518*
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
E__inference_dense_27_layer_call_and_return_conditional_losses_2450003x
IdentityIdentity)dense_27/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp/^batch_normalization_39/StatefulPartitionedCall/^batch_normalization_40/StatefulPartitionedCall/^batch_normalization_41/StatefulPartitionedCall"^conv2d_69/StatefulPartitionedCall"^conv2d_70/StatefulPartitionedCall"^conv2d_71/StatefulPartitionedCall"^conv2d_72/StatefulPartitionedCall"^conv2d_73/StatefulPartitionedCall!^dense_26/StatefulPartitionedCall!^dense_27/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:?????????8: : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_39/StatefulPartitionedCall.batch_normalization_39/StatefulPartitionedCall2`
.batch_normalization_40/StatefulPartitionedCall.batch_normalization_40/StatefulPartitionedCall2`
.batch_normalization_41/StatefulPartitionedCall.batch_normalization_41/StatefulPartitionedCall2F
!conv2d_69/StatefulPartitionedCall!conv2d_69/StatefulPartitionedCall2F
!conv2d_70/StatefulPartitionedCall!conv2d_70/StatefulPartitionedCall2F
!conv2d_71/StatefulPartitionedCall!conv2d_71/StatefulPartitionedCall2F
!conv2d_72/StatefulPartitionedCall!conv2d_72/StatefulPartitionedCall2F
!conv2d_73/StatefulPartitionedCall!conv2d_73/StatefulPartitionedCall2D
 dense_26/StatefulPartitionedCall dense_26/StatefulPartitionedCall2D
 dense_27/StatefulPartitionedCall dense_27/StatefulPartitionedCall:` \
/
_output_shapes
:?????????8
)
_user_specified_nameconv2d_69_input
?
?
F__inference_conv2d_71_layer_call_and_return_conditional_losses_2449907

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
?
?
S__inference_batch_normalization_39_layer_call_and_return_conditional_losses_2451013

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
?
?
S__inference_batch_normalization_39_layer_call_and_return_conditional_losses_2449682

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
i
M__inference_max_pooling2d_39_layer_call_and_return_conditional_losses_2449626

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
?
?
*__inference_dense_27_layer_call_fn_2451257

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
E__inference_dense_27_layer_call_and_return_conditional_losses_2450003o
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
?
?
S__inference_batch_normalization_41_layer_call_and_return_conditional_losses_2451199

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
?
?
S__inference_batch_normalization_40_layer_call_and_return_conditional_losses_2449727

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
?
?
F__inference_conv2d_73_layer_call_and_return_conditional_losses_2451145

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
??
?
J__inference_sequential_15_layer_call_and_return_conditional_losses_2450901

inputsB
(conv2d_69_conv2d_readvariableop_resource:@7
)conv2d_69_biasadd_readvariableop_resource:@B
(conv2d_70_conv2d_readvariableop_resource:@@7
)conv2d_70_biasadd_readvariableop_resource:@<
.batch_normalization_39_readvariableop_resource:@>
0batch_normalization_39_readvariableop_1_resource:@M
?batch_normalization_39_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_39_fusedbatchnormv3_readvariableop_1_resource:@C
(conv2d_71_conv2d_readvariableop_resource:@?8
)conv2d_71_biasadd_readvariableop_resource:	?D
(conv2d_72_conv2d_readvariableop_resource:??8
)conv2d_72_biasadd_readvariableop_resource:	?=
.batch_normalization_40_readvariableop_resource:	??
0batch_normalization_40_readvariableop_1_resource:	?N
?batch_normalization_40_fusedbatchnormv3_readvariableop_resource:	?P
Abatch_normalization_40_fusedbatchnormv3_readvariableop_1_resource:	?D
(conv2d_73_conv2d_readvariableop_resource:??8
)conv2d_73_biasadd_readvariableop_resource:	?=
.batch_normalization_41_readvariableop_resource:	??
0batch_normalization_41_readvariableop_1_resource:	?N
?batch_normalization_41_fusedbatchnormv3_readvariableop_resource:	?P
Abatch_normalization_41_fusedbatchnormv3_readvariableop_1_resource:	?;
'dense_26_matmul_readvariableop_resource:
??7
(dense_26_biasadd_readvariableop_resource:	?:
'dense_27_matmul_readvariableop_resource:	?6
(dense_27_biasadd_readvariableop_resource:
identity??%batch_normalization_39/AssignNewValue?'batch_normalization_39/AssignNewValue_1?6batch_normalization_39/FusedBatchNormV3/ReadVariableOp?8batch_normalization_39/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_39/ReadVariableOp?'batch_normalization_39/ReadVariableOp_1?%batch_normalization_40/AssignNewValue?'batch_normalization_40/AssignNewValue_1?6batch_normalization_40/FusedBatchNormV3/ReadVariableOp?8batch_normalization_40/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_40/ReadVariableOp?'batch_normalization_40/ReadVariableOp_1?%batch_normalization_41/AssignNewValue?'batch_normalization_41/AssignNewValue_1?6batch_normalization_41/FusedBatchNormV3/ReadVariableOp?8batch_normalization_41/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_41/ReadVariableOp?'batch_normalization_41/ReadVariableOp_1? conv2d_69/BiasAdd/ReadVariableOp?conv2d_69/Conv2D/ReadVariableOp? conv2d_70/BiasAdd/ReadVariableOp?conv2d_70/Conv2D/ReadVariableOp? conv2d_71/BiasAdd/ReadVariableOp?conv2d_71/Conv2D/ReadVariableOp? conv2d_72/BiasAdd/ReadVariableOp?conv2d_72/Conv2D/ReadVariableOp? conv2d_73/BiasAdd/ReadVariableOp?conv2d_73/Conv2D/ReadVariableOp?dense_26/BiasAdd/ReadVariableOp?dense_26/MatMul/ReadVariableOp?dense_27/BiasAdd/ReadVariableOp?dense_27/MatMul/ReadVariableOp?
conv2d_69/Conv2D/ReadVariableOpReadVariableOp(conv2d_69_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
conv2d_69/Conv2DConv2Dinputs'conv2d_69/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????6@*
paddingVALID*
strides
?
 conv2d_69/BiasAdd/ReadVariableOpReadVariableOp)conv2d_69_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_69/BiasAddBiasAddconv2d_69/Conv2D:output:0(conv2d_69/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????6@l
conv2d_69/ReluReluconv2d_69/BiasAdd:output:0*
T0*/
_output_shapes
:?????????6@?
conv2d_70/Conv2D/ReadVariableOpReadVariableOp(conv2d_70_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
conv2d_70/Conv2DConv2Dconv2d_69/Relu:activations:0'conv2d_70/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????4@*
paddingVALID*
strides
?
 conv2d_70/BiasAdd/ReadVariableOpReadVariableOp)conv2d_70_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_70/BiasAddBiasAddconv2d_70/Conv2D:output:0(conv2d_70/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????4@l
conv2d_70/ReluReluconv2d_70/BiasAdd:output:0*
T0*/
_output_shapes
:?????????4@?
max_pooling2d_39/MaxPoolMaxPoolconv2d_70/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
?
%batch_normalization_39/ReadVariableOpReadVariableOp.batch_normalization_39_readvariableop_resource*
_output_shapes
:@*
dtype0?
'batch_normalization_39/ReadVariableOp_1ReadVariableOp0batch_normalization_39_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
6batch_normalization_39/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_39_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
8batch_normalization_39/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_39_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
'batch_normalization_39/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_39/MaxPool:output:0-batch_normalization_39/ReadVariableOp:value:0/batch_normalization_39/ReadVariableOp_1:value:0>batch_normalization_39/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_39/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
exponential_avg_factor%
?#<?
%batch_normalization_39/AssignNewValueAssignVariableOp?batch_normalization_39_fusedbatchnormv3_readvariableop_resource4batch_normalization_39/FusedBatchNormV3:batch_mean:07^batch_normalization_39/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
'batch_normalization_39/AssignNewValue_1AssignVariableOpAbatch_normalization_39_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_39/FusedBatchNormV3:batch_variance:09^batch_normalization_39/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(?
conv2d_71/Conv2D/ReadVariableOpReadVariableOp(conv2d_71_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
conv2d_71/Conv2DConv2D+batch_normalization_39/FusedBatchNormV3:y:0'conv2d_71/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????
?*
paddingVALID*
strides
?
 conv2d_71/BiasAdd/ReadVariableOpReadVariableOp)conv2d_71_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
conv2d_71/BiasAddBiasAddconv2d_71/Conv2D:output:0(conv2d_71/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????
?m
conv2d_71/ReluReluconv2d_71/BiasAdd:output:0*
T0*0
_output_shapes
:?????????
??
conv2d_72/Conv2D/ReadVariableOpReadVariableOp(conv2d_72_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
conv2d_72/Conv2DConv2Dconv2d_71/Relu:activations:0'conv2d_72/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
?
 conv2d_72/BiasAdd/ReadVariableOpReadVariableOp)conv2d_72_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
conv2d_72/BiasAddBiasAddconv2d_72/Conv2D:output:0(conv2d_72/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????m
conv2d_72/ReluReluconv2d_72/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
max_pooling2d_40/MaxPoolMaxPoolconv2d_72/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
?
%batch_normalization_40/ReadVariableOpReadVariableOp.batch_normalization_40_readvariableop_resource*
_output_shapes	
:?*
dtype0?
'batch_normalization_40/ReadVariableOp_1ReadVariableOp0batch_normalization_40_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
6batch_normalization_40/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_40_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
8batch_normalization_40/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_40_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
'batch_normalization_40/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_40/MaxPool:output:0-batch_normalization_40/ReadVariableOp:value:0/batch_normalization_40/ReadVariableOp_1:value:0>batch_normalization_40/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_40/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
%batch_normalization_40/AssignNewValueAssignVariableOp?batch_normalization_40_fusedbatchnormv3_readvariableop_resource4batch_normalization_40/FusedBatchNormV3:batch_mean:07^batch_normalization_40/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
'batch_normalization_40/AssignNewValue_1AssignVariableOpAbatch_normalization_40_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_40/FusedBatchNormV3:batch_variance:09^batch_normalization_40/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(?
conv2d_73/Conv2D/ReadVariableOpReadVariableOp(conv2d_73_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
conv2d_73/Conv2DConv2D+batch_normalization_40/FusedBatchNormV3:y:0'conv2d_73/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	?*
paddingVALID*
strides
?
 conv2d_73/BiasAdd/ReadVariableOpReadVariableOp)conv2d_73_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
conv2d_73/BiasAddBiasAddconv2d_73/Conv2D:output:0(conv2d_73/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	?m
conv2d_73/ReluReluconv2d_73/BiasAdd:output:0*
T0*0
_output_shapes
:?????????	??
max_pooling2d_41/MaxPoolMaxPoolconv2d_73/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
?
%batch_normalization_41/ReadVariableOpReadVariableOp.batch_normalization_41_readvariableop_resource*
_output_shapes	
:?*
dtype0?
'batch_normalization_41/ReadVariableOp_1ReadVariableOp0batch_normalization_41_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
6batch_normalization_41/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_41_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
8batch_normalization_41/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_41_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
'batch_normalization_41/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_41/MaxPool:output:0-batch_normalization_41/ReadVariableOp:value:0/batch_normalization_41/ReadVariableOp_1:value:0>batch_normalization_41/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_41/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
exponential_avg_factor%
?#<?
%batch_normalization_41/AssignNewValueAssignVariableOp?batch_normalization_41_fusedbatchnormv3_readvariableop_resource4batch_normalization_41/FusedBatchNormV3:batch_mean:07^batch_normalization_41/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(?
'batch_normalization_41/AssignNewValue_1AssignVariableOpAbatch_normalization_41_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_41/FusedBatchNormV3:batch_variance:09^batch_normalization_41/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(a
flatten_13/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ?
flatten_13/ReshapeReshape+batch_normalization_41/FusedBatchNormV3:y:0flatten_13/Const:output:0*
T0*(
_output_shapes
:???????????
dense_26/MatMul/ReadVariableOpReadVariableOp'dense_26_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense_26/MatMulMatMulflatten_13/Reshape:output:0&dense_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_26/BiasAdd/ReadVariableOpReadVariableOp(dense_26_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_26/BiasAddBiasAdddense_26/MatMul:product:0'dense_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????c
dense_26/ReluReludense_26/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
dense_27/MatMul/ReadVariableOpReadVariableOp'dense_27_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
dense_27/MatMulMatMuldense_26/Relu:activations:0&dense_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_27/BiasAdd/ReadVariableOpReadVariableOp(dense_27_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_27/BiasAddBiasAdddense_27/MatMul:product:0'dense_27/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????h
dense_27/SoftmaxSoftmaxdense_27/BiasAdd:output:0*
T0*'
_output_shapes
:?????????i
IdentityIdentitydense_27/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????

NoOpNoOp&^batch_normalization_39/AssignNewValue(^batch_normalization_39/AssignNewValue_17^batch_normalization_39/FusedBatchNormV3/ReadVariableOp9^batch_normalization_39/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_39/ReadVariableOp(^batch_normalization_39/ReadVariableOp_1&^batch_normalization_40/AssignNewValue(^batch_normalization_40/AssignNewValue_17^batch_normalization_40/FusedBatchNormV3/ReadVariableOp9^batch_normalization_40/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_40/ReadVariableOp(^batch_normalization_40/ReadVariableOp_1&^batch_normalization_41/AssignNewValue(^batch_normalization_41/AssignNewValue_17^batch_normalization_41/FusedBatchNormV3/ReadVariableOp9^batch_normalization_41/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_41/ReadVariableOp(^batch_normalization_41/ReadVariableOp_1!^conv2d_69/BiasAdd/ReadVariableOp ^conv2d_69/Conv2D/ReadVariableOp!^conv2d_70/BiasAdd/ReadVariableOp ^conv2d_70/Conv2D/ReadVariableOp!^conv2d_71/BiasAdd/ReadVariableOp ^conv2d_71/Conv2D/ReadVariableOp!^conv2d_72/BiasAdd/ReadVariableOp ^conv2d_72/Conv2D/ReadVariableOp!^conv2d_73/BiasAdd/ReadVariableOp ^conv2d_73/Conv2D/ReadVariableOp ^dense_26/BiasAdd/ReadVariableOp^dense_26/MatMul/ReadVariableOp ^dense_27/BiasAdd/ReadVariableOp^dense_27/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:?????????8: : : : : : : : : : : : : : : : : : : : : : : : : : 2N
%batch_normalization_39/AssignNewValue%batch_normalization_39/AssignNewValue2R
'batch_normalization_39/AssignNewValue_1'batch_normalization_39/AssignNewValue_12p
6batch_normalization_39/FusedBatchNormV3/ReadVariableOp6batch_normalization_39/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_39/FusedBatchNormV3/ReadVariableOp_18batch_normalization_39/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_39/ReadVariableOp%batch_normalization_39/ReadVariableOp2R
'batch_normalization_39/ReadVariableOp_1'batch_normalization_39/ReadVariableOp_12N
%batch_normalization_40/AssignNewValue%batch_normalization_40/AssignNewValue2R
'batch_normalization_40/AssignNewValue_1'batch_normalization_40/AssignNewValue_12p
6batch_normalization_40/FusedBatchNormV3/ReadVariableOp6batch_normalization_40/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_40/FusedBatchNormV3/ReadVariableOp_18batch_normalization_40/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_40/ReadVariableOp%batch_normalization_40/ReadVariableOp2R
'batch_normalization_40/ReadVariableOp_1'batch_normalization_40/ReadVariableOp_12N
%batch_normalization_41/AssignNewValue%batch_normalization_41/AssignNewValue2R
'batch_normalization_41/AssignNewValue_1'batch_normalization_41/AssignNewValue_12p
6batch_normalization_41/FusedBatchNormV3/ReadVariableOp6batch_normalization_41/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_41/FusedBatchNormV3/ReadVariableOp_18batch_normalization_41/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_41/ReadVariableOp%batch_normalization_41/ReadVariableOp2R
'batch_normalization_41/ReadVariableOp_1'batch_normalization_41/ReadVariableOp_12D
 conv2d_69/BiasAdd/ReadVariableOp conv2d_69/BiasAdd/ReadVariableOp2B
conv2d_69/Conv2D/ReadVariableOpconv2d_69/Conv2D/ReadVariableOp2D
 conv2d_70/BiasAdd/ReadVariableOp conv2d_70/BiasAdd/ReadVariableOp2B
conv2d_70/Conv2D/ReadVariableOpconv2d_70/Conv2D/ReadVariableOp2D
 conv2d_71/BiasAdd/ReadVariableOp conv2d_71/BiasAdd/ReadVariableOp2B
conv2d_71/Conv2D/ReadVariableOpconv2d_71/Conv2D/ReadVariableOp2D
 conv2d_72/BiasAdd/ReadVariableOp conv2d_72/BiasAdd/ReadVariableOp2B
conv2d_72/Conv2D/ReadVariableOpconv2d_72/Conv2D/ReadVariableOp2D
 conv2d_73/BiasAdd/ReadVariableOp conv2d_73/BiasAdd/ReadVariableOp2B
conv2d_73/Conv2D/ReadVariableOpconv2d_73/Conv2D/ReadVariableOp2B
dense_26/BiasAdd/ReadVariableOpdense_26/BiasAdd/ReadVariableOp2@
dense_26/MatMul/ReadVariableOpdense_26/MatMul/ReadVariableOp2B
dense_27/BiasAdd/ReadVariableOpdense_27/BiasAdd/ReadVariableOp2@
dense_27/MatMul/ReadVariableOpdense_27/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????8
 
_user_specified_nameinputs
?
?
F__inference_conv2d_69_layer_call_and_return_conditional_losses_2450921

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
?
?
+__inference_conv2d_72_layer_call_fn_2451042

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
F__inference_conv2d_72_layer_call_and_return_conditional_losses_2449924x
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
?
H
,__inference_flatten_13_layer_call_fn_2451222

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
G__inference_flatten_13_layer_call_and_return_conditional_losses_2449973a
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
?
?
S__inference_batch_normalization_41_layer_call_and_return_conditional_losses_2449834

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
?	
?
8__inference_batch_normalization_40_layer_call_fn_2451089

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
S__inference_batch_normalization_40_layer_call_and_return_conditional_losses_2449758?
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
J__inference_sequential_15_layer_call_and_return_conditional_losses_2450801

inputsB
(conv2d_69_conv2d_readvariableop_resource:@7
)conv2d_69_biasadd_readvariableop_resource:@B
(conv2d_70_conv2d_readvariableop_resource:@@7
)conv2d_70_biasadd_readvariableop_resource:@<
.batch_normalization_39_readvariableop_resource:@>
0batch_normalization_39_readvariableop_1_resource:@M
?batch_normalization_39_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_39_fusedbatchnormv3_readvariableop_1_resource:@C
(conv2d_71_conv2d_readvariableop_resource:@?8
)conv2d_71_biasadd_readvariableop_resource:	?D
(conv2d_72_conv2d_readvariableop_resource:??8
)conv2d_72_biasadd_readvariableop_resource:	?=
.batch_normalization_40_readvariableop_resource:	??
0batch_normalization_40_readvariableop_1_resource:	?N
?batch_normalization_40_fusedbatchnormv3_readvariableop_resource:	?P
Abatch_normalization_40_fusedbatchnormv3_readvariableop_1_resource:	?D
(conv2d_73_conv2d_readvariableop_resource:??8
)conv2d_73_biasadd_readvariableop_resource:	?=
.batch_normalization_41_readvariableop_resource:	??
0batch_normalization_41_readvariableop_1_resource:	?N
?batch_normalization_41_fusedbatchnormv3_readvariableop_resource:	?P
Abatch_normalization_41_fusedbatchnormv3_readvariableop_1_resource:	?;
'dense_26_matmul_readvariableop_resource:
??7
(dense_26_biasadd_readvariableop_resource:	?:
'dense_27_matmul_readvariableop_resource:	?6
(dense_27_biasadd_readvariableop_resource:
identity??6batch_normalization_39/FusedBatchNormV3/ReadVariableOp?8batch_normalization_39/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_39/ReadVariableOp?'batch_normalization_39/ReadVariableOp_1?6batch_normalization_40/FusedBatchNormV3/ReadVariableOp?8batch_normalization_40/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_40/ReadVariableOp?'batch_normalization_40/ReadVariableOp_1?6batch_normalization_41/FusedBatchNormV3/ReadVariableOp?8batch_normalization_41/FusedBatchNormV3/ReadVariableOp_1?%batch_normalization_41/ReadVariableOp?'batch_normalization_41/ReadVariableOp_1? conv2d_69/BiasAdd/ReadVariableOp?conv2d_69/Conv2D/ReadVariableOp? conv2d_70/BiasAdd/ReadVariableOp?conv2d_70/Conv2D/ReadVariableOp? conv2d_71/BiasAdd/ReadVariableOp?conv2d_71/Conv2D/ReadVariableOp? conv2d_72/BiasAdd/ReadVariableOp?conv2d_72/Conv2D/ReadVariableOp? conv2d_73/BiasAdd/ReadVariableOp?conv2d_73/Conv2D/ReadVariableOp?dense_26/BiasAdd/ReadVariableOp?dense_26/MatMul/ReadVariableOp?dense_27/BiasAdd/ReadVariableOp?dense_27/MatMul/ReadVariableOp?
conv2d_69/Conv2D/ReadVariableOpReadVariableOp(conv2d_69_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0?
conv2d_69/Conv2DConv2Dinputs'conv2d_69/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????6@*
paddingVALID*
strides
?
 conv2d_69/BiasAdd/ReadVariableOpReadVariableOp)conv2d_69_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_69/BiasAddBiasAddconv2d_69/Conv2D:output:0(conv2d_69/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????6@l
conv2d_69/ReluReluconv2d_69/BiasAdd:output:0*
T0*/
_output_shapes
:?????????6@?
conv2d_70/Conv2D/ReadVariableOpReadVariableOp(conv2d_70_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0?
conv2d_70/Conv2DConv2Dconv2d_69/Relu:activations:0'conv2d_70/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????4@*
paddingVALID*
strides
?
 conv2d_70/BiasAdd/ReadVariableOpReadVariableOp)conv2d_70_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0?
conv2d_70/BiasAddBiasAddconv2d_70/Conv2D:output:0(conv2d_70/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????4@l
conv2d_70/ReluReluconv2d_70/BiasAdd:output:0*
T0*/
_output_shapes
:?????????4@?
max_pooling2d_39/MaxPoolMaxPoolconv2d_70/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
?
%batch_normalization_39/ReadVariableOpReadVariableOp.batch_normalization_39_readvariableop_resource*
_output_shapes
:@*
dtype0?
'batch_normalization_39/ReadVariableOp_1ReadVariableOp0batch_normalization_39_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
6batch_normalization_39/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_39_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0?
8batch_normalization_39/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_39_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0?
'batch_normalization_39/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_39/MaxPool:output:0-batch_normalization_39/ReadVariableOp:value:0/batch_normalization_39/ReadVariableOp_1:value:0>batch_normalization_39/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_39/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:?????????@:@:@:@:@:*
epsilon%o?:*
is_training( ?
conv2d_71/Conv2D/ReadVariableOpReadVariableOp(conv2d_71_conv2d_readvariableop_resource*'
_output_shapes
:@?*
dtype0?
conv2d_71/Conv2DConv2D+batch_normalization_39/FusedBatchNormV3:y:0'conv2d_71/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????
?*
paddingVALID*
strides
?
 conv2d_71/BiasAdd/ReadVariableOpReadVariableOp)conv2d_71_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
conv2d_71/BiasAddBiasAddconv2d_71/Conv2D:output:0(conv2d_71/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????
?m
conv2d_71/ReluReluconv2d_71/BiasAdd:output:0*
T0*0
_output_shapes
:?????????
??
conv2d_72/Conv2D/ReadVariableOpReadVariableOp(conv2d_72_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
conv2d_72/Conv2DConv2Dconv2d_71/Relu:activations:0'conv2d_72/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
?
 conv2d_72/BiasAdd/ReadVariableOpReadVariableOp)conv2d_72_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
conv2d_72/BiasAddBiasAddconv2d_72/Conv2D:output:0(conv2d_72/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????m
conv2d_72/ReluReluconv2d_72/BiasAdd:output:0*
T0*0
_output_shapes
:???????????
max_pooling2d_40/MaxPoolMaxPoolconv2d_72/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
?
%batch_normalization_40/ReadVariableOpReadVariableOp.batch_normalization_40_readvariableop_resource*
_output_shapes	
:?*
dtype0?
'batch_normalization_40/ReadVariableOp_1ReadVariableOp0batch_normalization_40_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
6batch_normalization_40/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_40_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
8batch_normalization_40/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_40_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
'batch_normalization_40/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_40/MaxPool:output:0-batch_normalization_40/ReadVariableOp:value:0/batch_normalization_40/ReadVariableOp_1:value:0>batch_normalization_40/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_40/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( ?
conv2d_73/Conv2D/ReadVariableOpReadVariableOp(conv2d_73_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype0?
conv2d_73/Conv2DConv2D+batch_normalization_40/FusedBatchNormV3:y:0'conv2d_73/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	?*
paddingVALID*
strides
?
 conv2d_73/BiasAdd/ReadVariableOpReadVariableOp)conv2d_73_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
conv2d_73/BiasAddBiasAddconv2d_73/Conv2D:output:0(conv2d_73/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????	?m
conv2d_73/ReluReluconv2d_73/BiasAdd:output:0*
T0*0
_output_shapes
:?????????	??
max_pooling2d_41/MaxPoolMaxPoolconv2d_73/Relu:activations:0*0
_output_shapes
:??????????*
ksize
*
paddingVALID*
strides
?
%batch_normalization_41/ReadVariableOpReadVariableOp.batch_normalization_41_readvariableop_resource*
_output_shapes	
:?*
dtype0?
'batch_normalization_41/ReadVariableOp_1ReadVariableOp0batch_normalization_41_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
6batch_normalization_41/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_41_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:?*
dtype0?
8batch_normalization_41/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_41_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:?*
dtype0?
'batch_normalization_41/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_41/MaxPool:output:0-batch_normalization_41/ReadVariableOp:value:0/batch_normalization_41/ReadVariableOp_1:value:0>batch_normalization_41/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_41/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:??????????:?:?:?:?:*
epsilon%o?:*
is_training( a
flatten_13/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   ?
flatten_13/ReshapeReshape+batch_normalization_41/FusedBatchNormV3:y:0flatten_13/Const:output:0*
T0*(
_output_shapes
:???????????
dense_26/MatMul/ReadVariableOpReadVariableOp'dense_26_matmul_readvariableop_resource* 
_output_shapes
:
??*
dtype0?
dense_26/MatMulMatMulflatten_13/Reshape:output:0&dense_26/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:???????????
dense_26/BiasAdd/ReadVariableOpReadVariableOp(dense_26_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype0?
dense_26/BiasAddBiasAdddense_26/MatMul:product:0'dense_26/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????c
dense_26/ReluReludense_26/BiasAdd:output:0*
T0*(
_output_shapes
:???????????
dense_27/MatMul/ReadVariableOpReadVariableOp'dense_27_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype0?
dense_27/MatMulMatMuldense_26/Relu:activations:0&dense_27/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:??????????
dense_27/BiasAdd/ReadVariableOpReadVariableOp(dense_27_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0?
dense_27/BiasAddBiasAdddense_27/MatMul:product:0'dense_27/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????h
dense_27/SoftmaxSoftmaxdense_27/BiasAdd:output:0*
T0*'
_output_shapes
:?????????i
IdentityIdentitydense_27/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:??????????
NoOpNoOp7^batch_normalization_39/FusedBatchNormV3/ReadVariableOp9^batch_normalization_39/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_39/ReadVariableOp(^batch_normalization_39/ReadVariableOp_17^batch_normalization_40/FusedBatchNormV3/ReadVariableOp9^batch_normalization_40/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_40/ReadVariableOp(^batch_normalization_40/ReadVariableOp_17^batch_normalization_41/FusedBatchNormV3/ReadVariableOp9^batch_normalization_41/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_41/ReadVariableOp(^batch_normalization_41/ReadVariableOp_1!^conv2d_69/BiasAdd/ReadVariableOp ^conv2d_69/Conv2D/ReadVariableOp!^conv2d_70/BiasAdd/ReadVariableOp ^conv2d_70/Conv2D/ReadVariableOp!^conv2d_71/BiasAdd/ReadVariableOp ^conv2d_71/Conv2D/ReadVariableOp!^conv2d_72/BiasAdd/ReadVariableOp ^conv2d_72/Conv2D/ReadVariableOp!^conv2d_73/BiasAdd/ReadVariableOp ^conv2d_73/Conv2D/ReadVariableOp ^dense_26/BiasAdd/ReadVariableOp^dense_26/MatMul/ReadVariableOp ^dense_27/BiasAdd/ReadVariableOp^dense_27/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:?????????8: : : : : : : : : : : : : : : : : : : : : : : : : : 2p
6batch_normalization_39/FusedBatchNormV3/ReadVariableOp6batch_normalization_39/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_39/FusedBatchNormV3/ReadVariableOp_18batch_normalization_39/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_39/ReadVariableOp%batch_normalization_39/ReadVariableOp2R
'batch_normalization_39/ReadVariableOp_1'batch_normalization_39/ReadVariableOp_12p
6batch_normalization_40/FusedBatchNormV3/ReadVariableOp6batch_normalization_40/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_40/FusedBatchNormV3/ReadVariableOp_18batch_normalization_40/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_40/ReadVariableOp%batch_normalization_40/ReadVariableOp2R
'batch_normalization_40/ReadVariableOp_1'batch_normalization_40/ReadVariableOp_12p
6batch_normalization_41/FusedBatchNormV3/ReadVariableOp6batch_normalization_41/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_41/FusedBatchNormV3/ReadVariableOp_18batch_normalization_41/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_41/ReadVariableOp%batch_normalization_41/ReadVariableOp2R
'batch_normalization_41/ReadVariableOp_1'batch_normalization_41/ReadVariableOp_12D
 conv2d_69/BiasAdd/ReadVariableOp conv2d_69/BiasAdd/ReadVariableOp2B
conv2d_69/Conv2D/ReadVariableOpconv2d_69/Conv2D/ReadVariableOp2D
 conv2d_70/BiasAdd/ReadVariableOp conv2d_70/BiasAdd/ReadVariableOp2B
conv2d_70/Conv2D/ReadVariableOpconv2d_70/Conv2D/ReadVariableOp2D
 conv2d_71/BiasAdd/ReadVariableOp conv2d_71/BiasAdd/ReadVariableOp2B
conv2d_71/Conv2D/ReadVariableOpconv2d_71/Conv2D/ReadVariableOp2D
 conv2d_72/BiasAdd/ReadVariableOp conv2d_72/BiasAdd/ReadVariableOp2B
conv2d_72/Conv2D/ReadVariableOpconv2d_72/Conv2D/ReadVariableOp2D
 conv2d_73/BiasAdd/ReadVariableOp conv2d_73/BiasAdd/ReadVariableOp2B
conv2d_73/Conv2D/ReadVariableOpconv2d_73/Conv2D/ReadVariableOp2B
dense_26/BiasAdd/ReadVariableOpdense_26/BiasAdd/ReadVariableOp2@
dense_26/MatMul/ReadVariableOpdense_26/MatMul/ReadVariableOp2B
dense_27/BiasAdd/ReadVariableOpdense_27/BiasAdd/ReadVariableOp2@
dense_27/MatMul/ReadVariableOpdense_27/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????8
 
_user_specified_nameinputs
?
?
/__inference_sequential_15_layer_call_fn_2450382
conv2d_69_input!
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
StatefulPartitionedCallStatefulPartitionedCallconv2d_69_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
J__inference_sequential_15_layer_call_and_return_conditional_losses_2450270o
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
_user_specified_nameconv2d_69_input
?
?
S__inference_batch_normalization_41_layer_call_and_return_conditional_losses_2451217

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
?
?
F__inference_conv2d_71_layer_call_and_return_conditional_losses_2451033

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
?
?
S__inference_batch_normalization_40_layer_call_and_return_conditional_losses_2451125

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
conv2d_69_input@
!serving_default_conv2d_69_input:0?????????8<
dense_270
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
/__inference_sequential_15_layer_call_fn_2450065
/__inference_sequential_15_layer_call_fn_2450644
/__inference_sequential_15_layer_call_fn_2450701
/__inference_sequential_15_layer_call_fn_2450382?
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
J__inference_sequential_15_layer_call_and_return_conditional_losses_2450801
J__inference_sequential_15_layer_call_and_return_conditional_losses_2450901
J__inference_sequential_15_layer_call_and_return_conditional_losses_2450452
J__inference_sequential_15_layer_call_and_return_conditional_losses_2450522?
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
"__inference__wrapped_model_2449617conv2d_69_input"?
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
+__inference_conv2d_69_layer_call_fn_2450910?
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
F__inference_conv2d_69_layer_call_and_return_conditional_losses_2450921?
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
*:(@2conv2d_69/kernel
:@2conv2d_69/bias
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
+__inference_conv2d_70_layer_call_fn_2450930?
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
F__inference_conv2d_70_layer_call_and_return_conditional_losses_2450941?
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
*:(@@2conv2d_70/kernel
:@2conv2d_70/bias
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
2__inference_max_pooling2d_39_layer_call_fn_2450946?
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
M__inference_max_pooling2d_39_layer_call_and_return_conditional_losses_2450951?
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
8__inference_batch_normalization_39_layer_call_fn_2450964
8__inference_batch_normalization_39_layer_call_fn_2450977?
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
S__inference_batch_normalization_39_layer_call_and_return_conditional_losses_2450995
S__inference_batch_normalization_39_layer_call_and_return_conditional_losses_2451013?
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
*:(@2batch_normalization_39/gamma
):'@2batch_normalization_39/beta
2:0@ (2"batch_normalization_39/moving_mean
6:4@ (2&batch_normalization_39/moving_variance
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
+__inference_conv2d_71_layer_call_fn_2451022?
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
F__inference_conv2d_71_layer_call_and_return_conditional_losses_2451033?
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
+:)@?2conv2d_71/kernel
:?2conv2d_71/bias
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
+__inference_conv2d_72_layer_call_fn_2451042?
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
F__inference_conv2d_72_layer_call_and_return_conditional_losses_2451053?
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
,:*??2conv2d_72/kernel
:?2conv2d_72/bias
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
2__inference_max_pooling2d_40_layer_call_fn_2451058?
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
M__inference_max_pooling2d_40_layer_call_and_return_conditional_losses_2451063?
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
8__inference_batch_normalization_40_layer_call_fn_2451076
8__inference_batch_normalization_40_layer_call_fn_2451089?
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
S__inference_batch_normalization_40_layer_call_and_return_conditional_losses_2451107
S__inference_batch_normalization_40_layer_call_and_return_conditional_losses_2451125?
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
+:)?2batch_normalization_40/gamma
*:(?2batch_normalization_40/beta
3:1? (2"batch_normalization_40/moving_mean
7:5? (2&batch_normalization_40/moving_variance
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
+__inference_conv2d_73_layer_call_fn_2451134?
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
F__inference_conv2d_73_layer_call_and_return_conditional_losses_2451145?
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
,:*??2conv2d_73/kernel
:?2conv2d_73/bias
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
2__inference_max_pooling2d_41_layer_call_fn_2451150?
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
M__inference_max_pooling2d_41_layer_call_and_return_conditional_losses_2451155?
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
8__inference_batch_normalization_41_layer_call_fn_2451168
8__inference_batch_normalization_41_layer_call_fn_2451181?
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
S__inference_batch_normalization_41_layer_call_and_return_conditional_losses_2451199
S__inference_batch_normalization_41_layer_call_and_return_conditional_losses_2451217?
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
+:)?2batch_normalization_41/gamma
*:(?2batch_normalization_41/beta
3:1? (2"batch_normalization_41/moving_mean
7:5? (2&batch_normalization_41/moving_variance
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
,__inference_flatten_13_layer_call_fn_2451222?
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
G__inference_flatten_13_layer_call_and_return_conditional_losses_2451228?
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
*__inference_dense_26_layer_call_fn_2451237?
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
E__inference_dense_26_layer_call_and_return_conditional_losses_2451248?
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
??2dense_26/kernel
:?2dense_26/bias
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
*__inference_dense_27_layer_call_fn_2451257?
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
E__inference_dense_27_layer_call_and_return_conditional_losses_2451268?
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
": 	?2dense_27/kernel
:2dense_27/bias
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
/__inference_sequential_15_layer_call_fn_2450065conv2d_69_input"?
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
/__inference_sequential_15_layer_call_fn_2450644inputs"?
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
/__inference_sequential_15_layer_call_fn_2450701inputs"?
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
/__inference_sequential_15_layer_call_fn_2450382conv2d_69_input"?
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
J__inference_sequential_15_layer_call_and_return_conditional_losses_2450801inputs"?
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
J__inference_sequential_15_layer_call_and_return_conditional_losses_2450901inputs"?
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
J__inference_sequential_15_layer_call_and_return_conditional_losses_2450452conv2d_69_input"?
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
J__inference_sequential_15_layer_call_and_return_conditional_losses_2450522conv2d_69_input"?
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
%__inference_signature_wrapper_2450587conv2d_69_input"?
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
+__inference_conv2d_69_layer_call_fn_2450910inputs"?
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
F__inference_conv2d_69_layer_call_and_return_conditional_losses_2450921inputs"?
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
+__inference_conv2d_70_layer_call_fn_2450930inputs"?
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
F__inference_conv2d_70_layer_call_and_return_conditional_losses_2450941inputs"?
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
2__inference_max_pooling2d_39_layer_call_fn_2450946inputs"?
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
M__inference_max_pooling2d_39_layer_call_and_return_conditional_losses_2450951inputs"?
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
8__inference_batch_normalization_39_layer_call_fn_2450964inputs"?
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
8__inference_batch_normalization_39_layer_call_fn_2450977inputs"?
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
S__inference_batch_normalization_39_layer_call_and_return_conditional_losses_2450995inputs"?
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
S__inference_batch_normalization_39_layer_call_and_return_conditional_losses_2451013inputs"?
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
+__inference_conv2d_71_layer_call_fn_2451022inputs"?
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
F__inference_conv2d_71_layer_call_and_return_conditional_losses_2451033inputs"?
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
+__inference_conv2d_72_layer_call_fn_2451042inputs"?
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
F__inference_conv2d_72_layer_call_and_return_conditional_losses_2451053inputs"?
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
2__inference_max_pooling2d_40_layer_call_fn_2451058inputs"?
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
M__inference_max_pooling2d_40_layer_call_and_return_conditional_losses_2451063inputs"?
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
8__inference_batch_normalization_40_layer_call_fn_2451076inputs"?
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
8__inference_batch_normalization_40_layer_call_fn_2451089inputs"?
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
S__inference_batch_normalization_40_layer_call_and_return_conditional_losses_2451107inputs"?
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
S__inference_batch_normalization_40_layer_call_and_return_conditional_losses_2451125inputs"?
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
+__inference_conv2d_73_layer_call_fn_2451134inputs"?
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
F__inference_conv2d_73_layer_call_and_return_conditional_losses_2451145inputs"?
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
2__inference_max_pooling2d_41_layer_call_fn_2451150inputs"?
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
M__inference_max_pooling2d_41_layer_call_and_return_conditional_losses_2451155inputs"?
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
8__inference_batch_normalization_41_layer_call_fn_2451168inputs"?
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
8__inference_batch_normalization_41_layer_call_fn_2451181inputs"?
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
S__inference_batch_normalization_41_layer_call_and_return_conditional_losses_2451199inputs"?
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
S__inference_batch_normalization_41_layer_call_and_return_conditional_losses_2451217inputs"?
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
,__inference_flatten_13_layer_call_fn_2451222inputs"?
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
G__inference_flatten_13_layer_call_and_return_conditional_losses_2451228inputs"?
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
*__inference_dense_26_layer_call_fn_2451237inputs"?
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
E__inference_dense_26_layer_call_and_return_conditional_losses_2451248inputs"?
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
*__inference_dense_27_layer_call_fn_2451257inputs"?
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
E__inference_dense_27_layer_call_and_return_conditional_losses_2451268inputs"?
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
/:-@2Adam/conv2d_69/kernel/m
!:@2Adam/conv2d_69/bias/m
/:-@@2Adam/conv2d_70/kernel/m
!:@2Adam/conv2d_70/bias/m
/:-@2#Adam/batch_normalization_39/gamma/m
.:,@2"Adam/batch_normalization_39/beta/m
0:.@?2Adam/conv2d_71/kernel/m
": ?2Adam/conv2d_71/bias/m
1:/??2Adam/conv2d_72/kernel/m
": ?2Adam/conv2d_72/bias/m
0:.?2#Adam/batch_normalization_40/gamma/m
/:-?2"Adam/batch_normalization_40/beta/m
1:/??2Adam/conv2d_73/kernel/m
": ?2Adam/conv2d_73/bias/m
0:.?2#Adam/batch_normalization_41/gamma/m
/:-?2"Adam/batch_normalization_41/beta/m
(:&
??2Adam/dense_26/kernel/m
!:?2Adam/dense_26/bias/m
':%	?2Adam/dense_27/kernel/m
 :2Adam/dense_27/bias/m
/:-@2Adam/conv2d_69/kernel/v
!:@2Adam/conv2d_69/bias/v
/:-@@2Adam/conv2d_70/kernel/v
!:@2Adam/conv2d_70/bias/v
/:-@2#Adam/batch_normalization_39/gamma/v
.:,@2"Adam/batch_normalization_39/beta/v
0:.@?2Adam/conv2d_71/kernel/v
": ?2Adam/conv2d_71/bias/v
1:/??2Adam/conv2d_72/kernel/v
": ?2Adam/conv2d_72/bias/v
0:.?2#Adam/batch_normalization_40/gamma/v
/:-?2"Adam/batch_normalization_40/beta/v
1:/??2Adam/conv2d_73/kernel/v
": ?2Adam/conv2d_73/bias/v
0:.?2#Adam/batch_normalization_41/gamma/v
/:-?2"Adam/batch_normalization_41/beta/v
(:&
??2Adam/dense_26/kernel/v
!:?2Adam/dense_26/bias/v
':%	?2Adam/dense_27/kernel/v
 :2Adam/dense_27/bias/v?
"__inference__wrapped_model_2449617?'(789:ABJKZ[\]detuvw????@?=
6?3
1?.
conv2d_69_input?????????8
? "3?0
.
dense_27"?
dense_27??????????
S__inference_batch_normalization_39_layer_call_and_return_conditional_losses_2450995?789:M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "??<
5?2
0+???????????????????????????@
? ?
S__inference_batch_normalization_39_layer_call_and_return_conditional_losses_2451013?789:M?J
C?@
:?7
inputs+???????????????????????????@
p
? "??<
5?2
0+???????????????????????????@
? ?
8__inference_batch_normalization_39_layer_call_fn_2450964?789:M?J
C?@
:?7
inputs+???????????????????????????@
p 
? "2?/+???????????????????????????@?
8__inference_batch_normalization_39_layer_call_fn_2450977?789:M?J
C?@
:?7
inputs+???????????????????????????@
p
? "2?/+???????????????????????????@?
S__inference_batch_normalization_40_layer_call_and_return_conditional_losses_2451107?Z[\]N?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
S__inference_batch_normalization_40_layer_call_and_return_conditional_losses_2451125?Z[\]N?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
8__inference_batch_normalization_40_layer_call_fn_2451076?Z[\]N?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
8__inference_batch_normalization_40_layer_call_fn_2451089?Z[\]N?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
S__inference_batch_normalization_41_layer_call_and_return_conditional_losses_2451199?tuvwN?K
D?A
;?8
inputs,????????????????????????????
p 
? "@?=
6?3
0,????????????????????????????
? ?
S__inference_batch_normalization_41_layer_call_and_return_conditional_losses_2451217?tuvwN?K
D?A
;?8
inputs,????????????????????????????
p
? "@?=
6?3
0,????????????????????????????
? ?
8__inference_batch_normalization_41_layer_call_fn_2451168?tuvwN?K
D?A
;?8
inputs,????????????????????????????
p 
? "3?0,?????????????????????????????
8__inference_batch_normalization_41_layer_call_fn_2451181?tuvwN?K
D?A
;?8
inputs,????????????????????????????
p
? "3?0,?????????????????????????????
F__inference_conv2d_69_layer_call_and_return_conditional_losses_2450921l7?4
-?*
(?%
inputs?????????8
? "-?*
#? 
0?????????6@
? ?
+__inference_conv2d_69_layer_call_fn_2450910_7?4
-?*
(?%
inputs?????????8
? " ??????????6@?
F__inference_conv2d_70_layer_call_and_return_conditional_losses_2450941l'(7?4
-?*
(?%
inputs?????????6@
? "-?*
#? 
0?????????4@
? ?
+__inference_conv2d_70_layer_call_fn_2450930_'(7?4
-?*
(?%
inputs?????????6@
? " ??????????4@?
F__inference_conv2d_71_layer_call_and_return_conditional_losses_2451033mAB7?4
-?*
(?%
inputs?????????@
? ".?+
$?!
0?????????
?
? ?
+__inference_conv2d_71_layer_call_fn_2451022`AB7?4
-?*
(?%
inputs?????????@
? "!??????????
??
F__inference_conv2d_72_layer_call_and_return_conditional_losses_2451053nJK8?5
.?+
)?&
inputs?????????
?
? ".?+
$?!
0??????????
? ?
+__inference_conv2d_72_layer_call_fn_2451042aJK8?5
.?+
)?&
inputs?????????
?
? "!????????????
F__inference_conv2d_73_layer_call_and_return_conditional_losses_2451145nde8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0?????????	?
? ?
+__inference_conv2d_73_layer_call_fn_2451134ade8?5
.?+
)?&
inputs??????????
? "!??????????	??
E__inference_dense_26_layer_call_and_return_conditional_losses_2451248`??0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????
? ?
*__inference_dense_26_layer_call_fn_2451237S??0?-
&?#
!?
inputs??????????
? "????????????
E__inference_dense_27_layer_call_and_return_conditional_losses_2451268_??0?-
&?#
!?
inputs??????????
? "%?"
?
0?????????
? ?
*__inference_dense_27_layer_call_fn_2451257R??0?-
&?#
!?
inputs??????????
? "???????????
G__inference_flatten_13_layer_call_and_return_conditional_losses_2451228b8?5
.?+
)?&
inputs??????????
? "&?#
?
0??????????
? ?
,__inference_flatten_13_layer_call_fn_2451222U8?5
.?+
)?&
inputs??????????
? "????????????
M__inference_max_pooling2d_39_layer_call_and_return_conditional_losses_2450951?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
2__inference_max_pooling2d_39_layer_call_fn_2450946?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
M__inference_max_pooling2d_40_layer_call_and_return_conditional_losses_2451063?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
2__inference_max_pooling2d_40_layer_call_fn_2451058?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
M__inference_max_pooling2d_41_layer_call_and_return_conditional_losses_2451155?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
2__inference_max_pooling2d_41_layer_call_fn_2451150?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
J__inference_sequential_15_layer_call_and_return_conditional_losses_2450452?'(789:ABJKZ[\]detuvw????H?E
>?;
1?.
conv2d_69_input?????????8
p 

 
? "%?"
?
0?????????
? ?
J__inference_sequential_15_layer_call_and_return_conditional_losses_2450522?'(789:ABJKZ[\]detuvw????H?E
>?;
1?.
conv2d_69_input?????????8
p

 
? "%?"
?
0?????????
? ?
J__inference_sequential_15_layer_call_and_return_conditional_losses_2450801?'(789:ABJKZ[\]detuvw??????<
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
J__inference_sequential_15_layer_call_and_return_conditional_losses_2450901?'(789:ABJKZ[\]detuvw??????<
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
/__inference_sequential_15_layer_call_fn_2450065?'(789:ABJKZ[\]detuvw????H?E
>?;
1?.
conv2d_69_input?????????8
p 

 
? "???????????
/__inference_sequential_15_layer_call_fn_2450382?'(789:ABJKZ[\]detuvw????H?E
>?;
1?.
conv2d_69_input?????????8
p

 
? "???????????
/__inference_sequential_15_layer_call_fn_2450644{'(789:ABJKZ[\]detuvw??????<
5?2
(?%
inputs?????????8
p 

 
? "???????????
/__inference_sequential_15_layer_call_fn_2450701{'(789:ABJKZ[\]detuvw??????<
5?2
(?%
inputs?????????8
p

 
? "???????????
%__inference_signature_wrapper_2450587?'(789:ABJKZ[\]detuvw????S?P
? 
I?F
D
conv2d_69_input1?.
conv2d_69_input?????????8"3?0
.
dense_27"?
dense_27?????????