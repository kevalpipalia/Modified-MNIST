÷
­ý
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
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

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
û
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
epsilonfloat%·Ñ8"&
exponential_avg_factorfloat%  ?";
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

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

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 
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
dtypetype
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
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
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
Á
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
executor_typestring ¨
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

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.10.02v2.10.0-rc3-6-g359c3cdfc5f8®í

Adam/dense_13/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_13/bias/v
y
(Adam/dense_13/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_13/bias/v*
_output_shapes
:*
dtype0

Adam/dense_13/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*'
shared_nameAdam/dense_13/kernel/v

*Adam/dense_13/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_13/kernel/v*
_output_shapes
:	*
dtype0

Adam/dense_12/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_12/bias/v
z
(Adam/dense_12/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_12/bias/v*
_output_shapes	
:*
dtype0

Adam/dense_12/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameAdam/dense_12/kernel/v

*Adam/dense_12/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_12/kernel/v* 
_output_shapes
:
*
dtype0

"Adam/batch_normalization_20/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_20/beta/v

6Adam/batch_normalization_20/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_20/beta/v*
_output_shapes	
:*
dtype0

#Adam/batch_normalization_20/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_20/gamma/v

7Adam/batch_normalization_20/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_20/gamma/v*
_output_shapes	
:*
dtype0

Adam/conv2d_38/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_38/bias/v
|
)Adam/conv2d_38/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_38/bias/v*
_output_shapes	
:*
dtype0

Adam/conv2d_38/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_38/kernel/v

+Adam/conv2d_38/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_38/kernel/v*(
_output_shapes
:*
dtype0

"Adam/batch_normalization_19/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_19/beta/v

6Adam/batch_normalization_19/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_19/beta/v*
_output_shapes	
:*
dtype0

#Adam/batch_normalization_19/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_19/gamma/v

7Adam/batch_normalization_19/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_19/gamma/v*
_output_shapes	
:*
dtype0

Adam/conv2d_37/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_37/bias/v
|
)Adam/conv2d_37/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_37/bias/v*
_output_shapes	
:*
dtype0

Adam/conv2d_37/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_37/kernel/v

+Adam/conv2d_37/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_37/kernel/v*(
_output_shapes
:*
dtype0

Adam/conv2d_36/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_36/bias/v
|
)Adam/conv2d_36/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_36/bias/v*
_output_shapes	
:*
dtype0

Adam/conv2d_36/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/conv2d_36/kernel/v

+Adam/conv2d_36/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_36/kernel/v*'
_output_shapes
:@*
dtype0

"Adam/batch_normalization_18/beta/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_18/beta/v

6Adam/batch_normalization_18/beta/v/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_18/beta/v*
_output_shapes
:@*
dtype0

#Adam/batch_normalization_18/gamma/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/batch_normalization_18/gamma/v

7Adam/batch_normalization_18/gamma/v/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_18/gamma/v*
_output_shapes
:@*
dtype0

Adam/conv2d_35/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_35/bias/v
{
)Adam/conv2d_35/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_35/bias/v*
_output_shapes
:@*
dtype0

Adam/conv2d_35/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv2d_35/kernel/v

+Adam/conv2d_35/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_35/kernel/v*&
_output_shapes
:@@*
dtype0

Adam/conv2d_34/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_34/bias/v
{
)Adam/conv2d_34/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_34/bias/v*
_output_shapes
:@*
dtype0

Adam/conv2d_34/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/conv2d_34/kernel/v

+Adam/conv2d_34/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d_34/kernel/v*&
_output_shapes
:@*
dtype0

Adam/dense_13/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_13/bias/m
y
(Adam/dense_13/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_13/bias/m*
_output_shapes
:*
dtype0

Adam/dense_13/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*'
shared_nameAdam/dense_13/kernel/m

*Adam/dense_13/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_13/kernel/m*
_output_shapes
:	*
dtype0

Adam/dense_12/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*%
shared_nameAdam/dense_12/bias/m
z
(Adam/dense_12/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_12/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_12/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*'
shared_nameAdam/dense_12/kernel/m

*Adam/dense_12/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_12/kernel/m* 
_output_shapes
:
*
dtype0

"Adam/batch_normalization_20/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_20/beta/m

6Adam/batch_normalization_20/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_20/beta/m*
_output_shapes	
:*
dtype0

#Adam/batch_normalization_20/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_20/gamma/m

7Adam/batch_normalization_20/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_20/gamma/m*
_output_shapes	
:*
dtype0

Adam/conv2d_38/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_38/bias/m
|
)Adam/conv2d_38/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_38/bias/m*
_output_shapes	
:*
dtype0

Adam/conv2d_38/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_38/kernel/m

+Adam/conv2d_38/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_38/kernel/m*(
_output_shapes
:*
dtype0

"Adam/batch_normalization_19/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"Adam/batch_normalization_19/beta/m

6Adam/batch_normalization_19/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_19/beta/m*
_output_shapes	
:*
dtype0

#Adam/batch_normalization_19/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#Adam/batch_normalization_19/gamma/m

7Adam/batch_normalization_19/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_19/gamma/m*
_output_shapes	
:*
dtype0

Adam/conv2d_37/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_37/bias/m
|
)Adam/conv2d_37/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_37/bias/m*
_output_shapes	
:*
dtype0

Adam/conv2d_37/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameAdam/conv2d_37/kernel/m

+Adam/conv2d_37/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_37/kernel/m*(
_output_shapes
:*
dtype0

Adam/conv2d_36/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_nameAdam/conv2d_36/bias/m
|
)Adam/conv2d_36/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_36/bias/m*
_output_shapes	
:*
dtype0

Adam/conv2d_36/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/conv2d_36/kernel/m

+Adam/conv2d_36/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_36/kernel/m*'
_output_shapes
:@*
dtype0

"Adam/batch_normalization_18/beta/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"Adam/batch_normalization_18/beta/m

6Adam/batch_normalization_18/beta/m/Read/ReadVariableOpReadVariableOp"Adam/batch_normalization_18/beta/m*
_output_shapes
:@*
dtype0

#Adam/batch_normalization_18/gamma/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*4
shared_name%#Adam/batch_normalization_18/gamma/m

7Adam/batch_normalization_18/gamma/m/Read/ReadVariableOpReadVariableOp#Adam/batch_normalization_18/gamma/m*
_output_shapes
:@*
dtype0

Adam/conv2d_35/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_35/bias/m
{
)Adam/conv2d_35/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_35/bias/m*
_output_shapes
:@*
dtype0

Adam/conv2d_35/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*(
shared_nameAdam/conv2d_35/kernel/m

+Adam/conv2d_35/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_35/kernel/m*&
_output_shapes
:@@*
dtype0

Adam/conv2d_34/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*&
shared_nameAdam/conv2d_34/bias/m
{
)Adam/conv2d_34/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_34/bias/m*
_output_shapes
:@*
dtype0

Adam/conv2d_34/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/conv2d_34/kernel/m

+Adam/conv2d_34/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d_34/kernel/m*&
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
dense_13/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_13/bias
k
!dense_13/bias/Read/ReadVariableOpReadVariableOpdense_13/bias*
_output_shapes
:*
dtype0
{
dense_13/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	* 
shared_namedense_13/kernel
t
#dense_13/kernel/Read/ReadVariableOpReadVariableOpdense_13/kernel*
_output_shapes
:	*
dtype0
s
dense_12/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_12/bias
l
!dense_12/bias/Read/ReadVariableOpReadVariableOpdense_12/bias*
_output_shapes	
:*
dtype0
|
dense_12/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
* 
shared_namedense_12/kernel
u
#dense_12/kernel/Read/ReadVariableOpReadVariableOpdense_12/kernel* 
_output_shapes
:
*
dtype0
¥
&batch_normalization_20/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_20/moving_variance

:batch_normalization_20/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_20/moving_variance*
_output_shapes	
:*
dtype0

"batch_normalization_20/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_20/moving_mean

6batch_normalization_20/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_20/moving_mean*
_output_shapes	
:*
dtype0

batch_normalization_20/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_20/beta

/batch_normalization_20/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_20/beta*
_output_shapes	
:*
dtype0

batch_normalization_20/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_20/gamma

0batch_normalization_20/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_20/gamma*
_output_shapes	
:*
dtype0
u
conv2d_38/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_38/bias
n
"conv2d_38/bias/Read/ReadVariableOpReadVariableOpconv2d_38/bias*
_output_shapes	
:*
dtype0

conv2d_38/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_38/kernel

$conv2d_38/kernel/Read/ReadVariableOpReadVariableOpconv2d_38/kernel*(
_output_shapes
:*
dtype0
¥
&batch_normalization_19/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*7
shared_name(&batch_normalization_19/moving_variance

:batch_normalization_19/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_19/moving_variance*
_output_shapes	
:*
dtype0

"batch_normalization_19/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*3
shared_name$"batch_normalization_19/moving_mean

6batch_normalization_19/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_19/moving_mean*
_output_shapes	
:*
dtype0

batch_normalization_19/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*,
shared_namebatch_normalization_19/beta

/batch_normalization_19/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_19/beta*
_output_shapes	
:*
dtype0

batch_normalization_19/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:*-
shared_namebatch_normalization_19/gamma

0batch_normalization_19/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_19/gamma*
_output_shapes	
:*
dtype0
u
conv2d_37/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_37/bias
n
"conv2d_37/bias/Read/ReadVariableOpReadVariableOpconv2d_37/bias*
_output_shapes	
:*
dtype0

conv2d_37/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:*!
shared_nameconv2d_37/kernel

$conv2d_37/kernel/Read/ReadVariableOpReadVariableOpconv2d_37/kernel*(
_output_shapes
:*
dtype0
u
conv2d_36/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_36/bias
n
"conv2d_36/bias/Read/ReadVariableOpReadVariableOpconv2d_36/bias*
_output_shapes	
:*
dtype0

conv2d_36/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameconv2d_36/kernel
~
$conv2d_36/kernel/Read/ReadVariableOpReadVariableOpconv2d_36/kernel*'
_output_shapes
:@*
dtype0
¤
&batch_normalization_18/moving_varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&batch_normalization_18/moving_variance

:batch_normalization_18/moving_variance/Read/ReadVariableOpReadVariableOp&batch_normalization_18/moving_variance*
_output_shapes
:@*
dtype0

"batch_normalization_18/moving_meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*3
shared_name$"batch_normalization_18/moving_mean

6batch_normalization_18/moving_mean/Read/ReadVariableOpReadVariableOp"batch_normalization_18/moving_mean*
_output_shapes
:@*
dtype0

batch_normalization_18/betaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*,
shared_namebatch_normalization_18/beta

/batch_normalization_18/beta/Read/ReadVariableOpReadVariableOpbatch_normalization_18/beta*
_output_shapes
:@*
dtype0

batch_normalization_18/gammaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*-
shared_namebatch_normalization_18/gamma

0batch_normalization_18/gamma/Read/ReadVariableOpReadVariableOpbatch_normalization_18/gamma*
_output_shapes
:@*
dtype0
t
conv2d_35/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_35/bias
m
"conv2d_35/bias/Read/ReadVariableOpReadVariableOpconv2d_35/bias*
_output_shapes
:@*
dtype0

conv2d_35/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameconv2d_35/kernel
}
$conv2d_35/kernel/Read/ReadVariableOpReadVariableOpconv2d_35/kernel*&
_output_shapes
:@@*
dtype0
t
conv2d_34/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_34/bias
m
"conv2d_34/bias/Read/ReadVariableOpReadVariableOpconv2d_34/bias*
_output_shapes
:@*
dtype0

conv2d_34/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameconv2d_34/kernel
}
$conv2d_34/kernel/Read/ReadVariableOpReadVariableOpconv2d_34/kernel*&
_output_shapes
:@*
dtype0

serving_default_conv2d_34_inputPlaceholder*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8*
dtype0*$
shape:ÿÿÿÿÿÿÿÿÿ8
ñ
StatefulPartitionedCallStatefulPartitionedCallserving_default_conv2d_34_inputconv2d_34/kernelconv2d_34/biasconv2d_35/kernelconv2d_35/biasbatch_normalization_18/gammabatch_normalization_18/beta"batch_normalization_18/moving_mean&batch_normalization_18/moving_varianceconv2d_36/kernelconv2d_36/biasconv2d_37/kernelconv2d_37/biasbatch_normalization_19/gammabatch_normalization_19/beta"batch_normalization_19/moving_mean&batch_normalization_19/moving_varianceconv2d_38/kernelconv2d_38/biasbatch_normalization_20/gammabatch_normalization_20/beta"batch_normalization_20/moving_mean&batch_normalization_20/moving_variancedense_12/kerneldense_12/biasdense_13/kerneldense_13/bias*&
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *.
f)R'
%__inference_signature_wrapper_2432695

NoOpNoOp
 
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ú
valueÏBË BÃ

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
È
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses

kernel
bias
  _jit_compiled_convolution_op*
È
!	variables
"trainable_variables
#regularization_losses
$	keras_api
%__call__
*&&call_and_return_all_conditional_losses

'kernel
(bias
 )_jit_compiled_convolution_op*

*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses* 
Õ
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
È
;	variables
<trainable_variables
=regularization_losses
>	keras_api
?__call__
*@&call_and_return_all_conditional_losses

Akernel
Bbias
 C_jit_compiled_convolution_op*
È
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
H__call__
*I&call_and_return_all_conditional_losses

Jkernel
Kbias
 L_jit_compiled_convolution_op*

M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses* 
Õ
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
È
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses

dkernel
ebias
 f_jit_compiled_convolution_op*

g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses* 
Õ
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

x	variables
ytrainable_variables
zregularization_losses
{	keras_api
|__call__
*}&call_and_return_all_conditional_losses* 
¬
~	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias*
®
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias*
Î
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
22
23
24
25*

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
16
17
18
19*
* 
µ
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
:
trace_0
trace_1
trace_2
trace_3* 
:
trace_0
trace_1
trace_2
trace_3* 
* 
á
	iter
beta_1
beta_2

decay
learning_ratemm'm(m7m8mAmBmJmKmZm[mdm em¡tm¢um£	m¤	m¥	m¦	m§v¨v©'vª(v«7v¬8v­Av®Bv¯Jv°Kv±Zv²[v³dv´evµtv¶uv·	v¸	v¹	vº	v»*

 serving_default* 

0
1*

0
1*
* 

¡non_trainable_variables
¢layers
£metrics
 ¤layer_regularization_losses
¥layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

¦trace_0* 

§trace_0* 
`Z
VARIABLE_VALUEconv2d_34/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_34/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

'0
(1*

'0
(1*
* 

¨non_trainable_variables
©layers
ªmetrics
 «layer_regularization_losses
¬layer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses*

­trace_0* 

®trace_0* 
`Z
VARIABLE_VALUEconv2d_35/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_35/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

¯non_trainable_variables
°layers
±metrics
 ²layer_regularization_losses
³layer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses* 

´trace_0* 

µtrace_0* 
 
70
81
92
:3*

70
81*
* 

¶non_trainable_variables
·layers
¸metrics
 ¹layer_regularization_losses
ºlayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses*

»trace_0
¼trace_1* 

½trace_0
¾trace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_18/gamma5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_18/beta4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_18/moving_mean;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_18/moving_variance?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*

A0
B1*

A0
B1*
* 

¿non_trainable_variables
Àlayers
Ámetrics
 Âlayer_regularization_losses
Ãlayer_metrics
;	variables
<trainable_variables
=regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses*

Ätrace_0* 

Åtrace_0* 
`Z
VARIABLE_VALUEconv2d_36/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_36/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 

J0
K1*

J0
K1*
* 

Ænon_trainable_variables
Çlayers
Èmetrics
 Élayer_regularization_losses
Êlayer_metrics
D	variables
Etrainable_variables
Fregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses*

Ëtrace_0* 

Ìtrace_0* 
`Z
VARIABLE_VALUEconv2d_37/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_37/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

Ínon_trainable_variables
Îlayers
Ïmetrics
 Ðlayer_regularization_losses
Ñlayer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses* 

Òtrace_0* 

Ótrace_0* 
 
Z0
[1
\2
]3*

Z0
[1*
* 

Ônon_trainable_variables
Õlayers
Ömetrics
 ×layer_regularization_losses
Ølayer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses*

Ùtrace_0
Útrace_1* 

Ûtrace_0
Ütrace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_19/gamma5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_19/beta4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_19/moving_mean;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_19/moving_variance?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*

d0
e1*

d0
e1*
* 

Ýnon_trainable_variables
Þlayers
ßmetrics
 àlayer_regularization_losses
álayer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses*

âtrace_0* 

ãtrace_0* 
`Z
VARIABLE_VALUEconv2d_38/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEconv2d_38/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
* 

änon_trainable_variables
ålayers
æmetrics
 çlayer_regularization_losses
èlayer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses* 

étrace_0* 

êtrace_0* 
 
t0
u1
v2
w3*

t0
u1*
* 

ënon_trainable_variables
ìlayers
ímetrics
 îlayer_regularization_losses
ïlayer_metrics
m	variables
ntrainable_variables
oregularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses*

ðtrace_0
ñtrace_1* 

òtrace_0
ótrace_1* 
* 
ke
VARIABLE_VALUEbatch_normalization_20/gamma5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUE*
ic
VARIABLE_VALUEbatch_normalization_20/beta4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUE*
wq
VARIABLE_VALUE"batch_normalization_20/moving_mean;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUE&batch_normalization_20/moving_variance?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

ônon_trainable_variables
õlayers
ömetrics
 ÷layer_regularization_losses
ølayer_metrics
x	variables
ytrainable_variables
zregularization_losses
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses* 

ùtrace_0* 

útrace_0* 

0
1*

0
1*
* 

ûnon_trainable_variables
ülayers
ýmetrics
 þlayer_regularization_losses
ÿlayer_metrics
~	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

trace_0* 

trace_0* 
_Y
VARIABLE_VALUEdense_12/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_12/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

trace_0* 

trace_0* 
_Y
VARIABLE_VALUEdense_13/kernel6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_13/bias4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
0
1*
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
	variables
	keras_api

total

count*
M
	variables
	keras_api

total

count

_fn_kwargs*

0
1*

	variables*
UO
VARIABLE_VALUEtotal_14keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

	variables*
SM
VARIABLE_VALUEtotal4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE*
* 
}
VARIABLE_VALUEAdam/conv2d_34/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_34/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/conv2d_35/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_35/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_18/gamma/mQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE"Adam/batch_normalization_18/beta/mPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/conv2d_36/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_36/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/conv2d_37/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_37/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_19/gamma/mQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE"Adam/batch_normalization_19/beta/mPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/conv2d_38/kernel/mRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_38/bias/mPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_20/gamma/mQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE"Adam/batch_normalization_20/beta/mPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_12/kernel/mRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_12/bias/mPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_13/kernel/mRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_13/bias/mPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/conv2d_34/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_34/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/conv2d_35/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_35/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_18/gamma/vQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE"Adam/batch_normalization_18/beta/vPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/conv2d_36/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_36/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/conv2d_37/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_37/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_19/gamma/vQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE"Adam/batch_normalization_19/beta/vPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
}
VARIABLE_VALUEAdam/conv2d_38/kernel/vRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
y
VARIABLE_VALUEAdam/conv2d_38/bias/vPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE#Adam/batch_normalization_20/gamma/vQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE"Adam/batch_normalization_20/beta/vPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_12/kernel/vRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_12/bias/vPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
VARIABLE_VALUEAdam/dense_13/kernel/vRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~x
VARIABLE_VALUEAdam/dense_13/bias/vPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ø
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_34/kernel/Read/ReadVariableOp"conv2d_34/bias/Read/ReadVariableOp$conv2d_35/kernel/Read/ReadVariableOp"conv2d_35/bias/Read/ReadVariableOp0batch_normalization_18/gamma/Read/ReadVariableOp/batch_normalization_18/beta/Read/ReadVariableOp6batch_normalization_18/moving_mean/Read/ReadVariableOp:batch_normalization_18/moving_variance/Read/ReadVariableOp$conv2d_36/kernel/Read/ReadVariableOp"conv2d_36/bias/Read/ReadVariableOp$conv2d_37/kernel/Read/ReadVariableOp"conv2d_37/bias/Read/ReadVariableOp0batch_normalization_19/gamma/Read/ReadVariableOp/batch_normalization_19/beta/Read/ReadVariableOp6batch_normalization_19/moving_mean/Read/ReadVariableOp:batch_normalization_19/moving_variance/Read/ReadVariableOp$conv2d_38/kernel/Read/ReadVariableOp"conv2d_38/bias/Read/ReadVariableOp0batch_normalization_20/gamma/Read/ReadVariableOp/batch_normalization_20/beta/Read/ReadVariableOp6batch_normalization_20/moving_mean/Read/ReadVariableOp:batch_normalization_20/moving_variance/Read/ReadVariableOp#dense_12/kernel/Read/ReadVariableOp!dense_12/bias/Read/ReadVariableOp#dense_13/kernel/Read/ReadVariableOp!dense_13/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp+Adam/conv2d_34/kernel/m/Read/ReadVariableOp)Adam/conv2d_34/bias/m/Read/ReadVariableOp+Adam/conv2d_35/kernel/m/Read/ReadVariableOp)Adam/conv2d_35/bias/m/Read/ReadVariableOp7Adam/batch_normalization_18/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_18/beta/m/Read/ReadVariableOp+Adam/conv2d_36/kernel/m/Read/ReadVariableOp)Adam/conv2d_36/bias/m/Read/ReadVariableOp+Adam/conv2d_37/kernel/m/Read/ReadVariableOp)Adam/conv2d_37/bias/m/Read/ReadVariableOp7Adam/batch_normalization_19/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_19/beta/m/Read/ReadVariableOp+Adam/conv2d_38/kernel/m/Read/ReadVariableOp)Adam/conv2d_38/bias/m/Read/ReadVariableOp7Adam/batch_normalization_20/gamma/m/Read/ReadVariableOp6Adam/batch_normalization_20/beta/m/Read/ReadVariableOp*Adam/dense_12/kernel/m/Read/ReadVariableOp(Adam/dense_12/bias/m/Read/ReadVariableOp*Adam/dense_13/kernel/m/Read/ReadVariableOp(Adam/dense_13/bias/m/Read/ReadVariableOp+Adam/conv2d_34/kernel/v/Read/ReadVariableOp)Adam/conv2d_34/bias/v/Read/ReadVariableOp+Adam/conv2d_35/kernel/v/Read/ReadVariableOp)Adam/conv2d_35/bias/v/Read/ReadVariableOp7Adam/batch_normalization_18/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_18/beta/v/Read/ReadVariableOp+Adam/conv2d_36/kernel/v/Read/ReadVariableOp)Adam/conv2d_36/bias/v/Read/ReadVariableOp+Adam/conv2d_37/kernel/v/Read/ReadVariableOp)Adam/conv2d_37/bias/v/Read/ReadVariableOp7Adam/batch_normalization_19/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_19/beta/v/Read/ReadVariableOp+Adam/conv2d_38/kernel/v/Read/ReadVariableOp)Adam/conv2d_38/bias/v/Read/ReadVariableOp7Adam/batch_normalization_20/gamma/v/Read/ReadVariableOp6Adam/batch_normalization_20/beta/v/Read/ReadVariableOp*Adam/dense_12/kernel/v/Read/ReadVariableOp(Adam/dense_12/bias/v/Read/ReadVariableOp*Adam/dense_13/kernel/v/Read/ReadVariableOp(Adam/dense_13/bias/v/Read/ReadVariableOpConst*X
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
GPU2*0J 8 *)
f$R"
 __inference__traced_save_2433624

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_34/kernelconv2d_34/biasconv2d_35/kernelconv2d_35/biasbatch_normalization_18/gammabatch_normalization_18/beta"batch_normalization_18/moving_mean&batch_normalization_18/moving_varianceconv2d_36/kernelconv2d_36/biasconv2d_37/kernelconv2d_37/biasbatch_normalization_19/gammabatch_normalization_19/beta"batch_normalization_19/moving_mean&batch_normalization_19/moving_varianceconv2d_38/kernelconv2d_38/biasbatch_normalization_20/gammabatch_normalization_20/beta"batch_normalization_20/moving_mean&batch_normalization_20/moving_variancedense_12/kerneldense_12/biasdense_13/kerneldense_13/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotal_1count_1totalcountAdam/conv2d_34/kernel/mAdam/conv2d_34/bias/mAdam/conv2d_35/kernel/mAdam/conv2d_35/bias/m#Adam/batch_normalization_18/gamma/m"Adam/batch_normalization_18/beta/mAdam/conv2d_36/kernel/mAdam/conv2d_36/bias/mAdam/conv2d_37/kernel/mAdam/conv2d_37/bias/m#Adam/batch_normalization_19/gamma/m"Adam/batch_normalization_19/beta/mAdam/conv2d_38/kernel/mAdam/conv2d_38/bias/m#Adam/batch_normalization_20/gamma/m"Adam/batch_normalization_20/beta/mAdam/dense_12/kernel/mAdam/dense_12/bias/mAdam/dense_13/kernel/mAdam/dense_13/bias/mAdam/conv2d_34/kernel/vAdam/conv2d_34/bias/vAdam/conv2d_35/kernel/vAdam/conv2d_35/bias/v#Adam/batch_normalization_18/gamma/v"Adam/batch_normalization_18/beta/vAdam/conv2d_36/kernel/vAdam/conv2d_36/bias/vAdam/conv2d_37/kernel/vAdam/conv2d_37/bias/v#Adam/batch_normalization_19/gamma/v"Adam/batch_normalization_19/beta/vAdam/conv2d_38/kernel/vAdam/conv2d_38/bias/v#Adam/batch_normalization_20/gamma/v"Adam/batch_normalization_20/beta/vAdam/dense_12/kernel/vAdam/dense_12/bias/vAdam/dense_13/kernel/vAdam/dense_13/bias/v*W
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
GPU2*0J 8 *,
f'R%
#__inference__traced_restore_2433859Ìý
ñ
 
+__inference_conv2d_34_layer_call_fn_2433018

inputs!
unknown:@
	unknown_0:@
identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ6@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_34_layer_call_and_return_conditional_losses_2431971w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ6@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ8: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8
 
_user_specified_nameinputs

Æ
S__inference_batch_normalization_20_layer_call_and_return_conditional_losses_2431942

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Û
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<Æ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ð
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÔ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ü
Ö
%__inference_signature_wrapper_2432695
conv2d_34_input!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@
	unknown_6:@$
	unknown_7:@
	unknown_8:	%
	unknown_9:

unknown_10:	

unknown_11:	

unknown_12:	

unknown_13:	

unknown_14:	&

unknown_15:

unknown_16:	

unknown_17:	

unknown_18:	

unknown_19:	

unknown_20:	

unknown_21:


unknown_22:	

unknown_23:	

unknown_24:
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallconv2d_34_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
:ÿÿÿÿÿÿÿÿÿ*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference__wrapped_model_2431725o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:ÿÿÿÿÿÿÿÿÿ8: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8
)
_user_specified_nameconv2d_34_input
ñ
 
+__inference_conv2d_35_layer_call_fn_2433038

inputs!
unknown:@@
	unknown_0:@
identity¢StatefulPartitionedCallæ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_35_layer_call_and_return_conditional_losses_2431988w
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ6@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ6@
 
_user_specified_nameinputs


F__inference_conv2d_37_layer_call_and_return_conditional_losses_2433161

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
Þ
¢
S__inference_batch_normalization_20_layer_call_and_return_conditional_losses_2433307

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Í
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

i
M__inference_max_pooling2d_20_layer_call_and_return_conditional_losses_2433263

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¦
ß
.__inference_sequential_8_layer_call_fn_2432490
conv2d_34_input!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@
	unknown_6:@$
	unknown_7:@
	unknown_8:	%
	unknown_9:

unknown_10:	

unknown_11:	

unknown_12:	

unknown_13:	

unknown_14:	&

unknown_15:

unknown_16:	

unknown_17:	

unknown_18:	

unknown_19:	

unknown_20:	

unknown_21:


unknown_22:	

unknown_23:	

unknown_24:
identity¢StatefulPartitionedCall«
StatefulPartitionedCallStatefulPartitionedCallconv2d_34_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
:ÿÿÿÿÿÿÿÿÿ*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_sequential_8_layer_call_and_return_conditional_losses_2432378o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:ÿÿÿÿÿÿÿÿÿ8: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8
)
_user_specified_nameconv2d_34_input
ø
£
+__inference_conv2d_38_layer_call_fn_2433242

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallç
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_38_layer_call_and_return_conditional_losses_2432059x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ñ
Ô
I__inference_sequential_8_layer_call_and_return_conditional_losses_2433009

inputsB
(conv2d_34_conv2d_readvariableop_resource:@7
)conv2d_34_biasadd_readvariableop_resource:@B
(conv2d_35_conv2d_readvariableop_resource:@@7
)conv2d_35_biasadd_readvariableop_resource:@<
.batch_normalization_18_readvariableop_resource:@>
0batch_normalization_18_readvariableop_1_resource:@M
?batch_normalization_18_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_18_fusedbatchnormv3_readvariableop_1_resource:@C
(conv2d_36_conv2d_readvariableop_resource:@8
)conv2d_36_biasadd_readvariableop_resource:	D
(conv2d_37_conv2d_readvariableop_resource:8
)conv2d_37_biasadd_readvariableop_resource:	=
.batch_normalization_19_readvariableop_resource:	?
0batch_normalization_19_readvariableop_1_resource:	N
?batch_normalization_19_fusedbatchnormv3_readvariableop_resource:	P
Abatch_normalization_19_fusedbatchnormv3_readvariableop_1_resource:	D
(conv2d_38_conv2d_readvariableop_resource:8
)conv2d_38_biasadd_readvariableop_resource:	=
.batch_normalization_20_readvariableop_resource:	?
0batch_normalization_20_readvariableop_1_resource:	N
?batch_normalization_20_fusedbatchnormv3_readvariableop_resource:	P
Abatch_normalization_20_fusedbatchnormv3_readvariableop_1_resource:	;
'dense_12_matmul_readvariableop_resource:
7
(dense_12_biasadd_readvariableop_resource:	:
'dense_13_matmul_readvariableop_resource:	6
(dense_13_biasadd_readvariableop_resource:
identity¢%batch_normalization_18/AssignNewValue¢'batch_normalization_18/AssignNewValue_1¢6batch_normalization_18/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_18/ReadVariableOp¢'batch_normalization_18/ReadVariableOp_1¢%batch_normalization_19/AssignNewValue¢'batch_normalization_19/AssignNewValue_1¢6batch_normalization_19/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_19/ReadVariableOp¢'batch_normalization_19/ReadVariableOp_1¢%batch_normalization_20/AssignNewValue¢'batch_normalization_20/AssignNewValue_1¢6batch_normalization_20/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_20/ReadVariableOp¢'batch_normalization_20/ReadVariableOp_1¢ conv2d_34/BiasAdd/ReadVariableOp¢conv2d_34/Conv2D/ReadVariableOp¢ conv2d_35/BiasAdd/ReadVariableOp¢conv2d_35/Conv2D/ReadVariableOp¢ conv2d_36/BiasAdd/ReadVariableOp¢conv2d_36/Conv2D/ReadVariableOp¢ conv2d_37/BiasAdd/ReadVariableOp¢conv2d_37/Conv2D/ReadVariableOp¢ conv2d_38/BiasAdd/ReadVariableOp¢conv2d_38/Conv2D/ReadVariableOp¢dense_12/BiasAdd/ReadVariableOp¢dense_12/MatMul/ReadVariableOp¢dense_13/BiasAdd/ReadVariableOp¢dense_13/MatMul/ReadVariableOp
conv2d_34/Conv2D/ReadVariableOpReadVariableOp(conv2d_34_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0®
conv2d_34/Conv2DConv2Dinputs'conv2d_34/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ6@*
paddingVALID*
strides

 conv2d_34/BiasAdd/ReadVariableOpReadVariableOp)conv2d_34_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_34/BiasAddBiasAddconv2d_34/Conv2D:output:0(conv2d_34/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ6@l
conv2d_34/ReluReluconv2d_34/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ6@
conv2d_35/Conv2D/ReadVariableOpReadVariableOp(conv2d_35_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Ä
conv2d_35/Conv2DConv2Dconv2d_34/Relu:activations:0'conv2d_35/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4@*
paddingVALID*
strides

 conv2d_35/BiasAdd/ReadVariableOpReadVariableOp)conv2d_35_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_35/BiasAddBiasAddconv2d_35/Conv2D:output:0(conv2d_35/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4@l
conv2d_35/ReluReluconv2d_35/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4@®
max_pooling2d_18/MaxPoolMaxPoolconv2d_35/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides

%batch_normalization_18/ReadVariableOpReadVariableOp.batch_normalization_18_readvariableop_resource*
_output_shapes
:@*
dtype0
'batch_normalization_18/ReadVariableOp_1ReadVariableOp0batch_normalization_18_readvariableop_1_resource*
_output_shapes
:@*
dtype0²
6batch_normalization_18/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_18_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0¶
8batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_18_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ò
'batch_normalization_18/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_18/MaxPool:output:0-batch_normalization_18/ReadVariableOp:value:0/batch_normalization_18/ReadVariableOp_1:value:0>batch_normalization_18/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<¢
%batch_normalization_18/AssignNewValueAssignVariableOp?batch_normalization_18_fusedbatchnormv3_readvariableop_resource4batch_normalization_18/FusedBatchNormV3:batch_mean:07^batch_normalization_18/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(¬
'batch_normalization_18/AssignNewValue_1AssignVariableOpAbatch_normalization_18_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_18/FusedBatchNormV3:batch_variance:09^batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(
conv2d_36/Conv2D/ReadVariableOpReadVariableOp(conv2d_36_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0Ô
conv2d_36/Conv2DConv2D+batch_normalization_18/FusedBatchNormV3:y:0'conv2d_36/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
paddingVALID*
strides

 conv2d_36/BiasAdd/ReadVariableOpReadVariableOp)conv2d_36_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_36/BiasAddBiasAddconv2d_36/Conv2D:output:0(conv2d_36/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
m
conv2d_36/ReluReluconv2d_36/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

conv2d_37/Conv2D/ReadVariableOpReadVariableOp(conv2d_37_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Å
conv2d_37/Conv2DConv2Dconv2d_36/Relu:activations:0'conv2d_37/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides

 conv2d_37/BiasAdd/ReadVariableOpReadVariableOp)conv2d_37_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_37/BiasAddBiasAddconv2d_37/Conv2D:output:0(conv2d_37/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
conv2d_37/ReluReluconv2d_37/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯
max_pooling2d_19/MaxPoolMaxPoolconv2d_37/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

%batch_normalization_19/ReadVariableOpReadVariableOp.batch_normalization_19_readvariableop_resource*
_output_shapes	
:*
dtype0
'batch_normalization_19/ReadVariableOp_1ReadVariableOp0batch_normalization_19_readvariableop_1_resource*
_output_shapes	
:*
dtype0³
6batch_normalization_19/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_19_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0·
8batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_19_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0×
'batch_normalization_19/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_19/MaxPool:output:0-batch_normalization_19/ReadVariableOp:value:0/batch_normalization_19/ReadVariableOp_1:value:0>batch_normalization_19/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<¢
%batch_normalization_19/AssignNewValueAssignVariableOp?batch_normalization_19_fusedbatchnormv3_readvariableop_resource4batch_normalization_19/FusedBatchNormV3:batch_mean:07^batch_normalization_19/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(¬
'batch_normalization_19/AssignNewValue_1AssignVariableOpAbatch_normalization_19_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_19/FusedBatchNormV3:batch_variance:09^batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(
conv2d_38/Conv2D/ReadVariableOpReadVariableOp(conv2d_38_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ô
conv2d_38/Conv2DConv2D+batch_normalization_19/FusedBatchNormV3:y:0'conv2d_38/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	*
paddingVALID*
strides

 conv2d_38/BiasAdd/ReadVariableOpReadVariableOp)conv2d_38_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_38/BiasAddBiasAddconv2d_38/Conv2D:output:0(conv2d_38/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	m
conv2d_38/ReluReluconv2d_38/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	¯
max_pooling2d_20/MaxPoolMaxPoolconv2d_38/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

%batch_normalization_20/ReadVariableOpReadVariableOp.batch_normalization_20_readvariableop_resource*
_output_shapes	
:*
dtype0
'batch_normalization_20/ReadVariableOp_1ReadVariableOp0batch_normalization_20_readvariableop_1_resource*
_output_shapes	
:*
dtype0³
6batch_normalization_20/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_20_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0·
8batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_20_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0×
'batch_normalization_20/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_20/MaxPool:output:0-batch_normalization_20/ReadVariableOp:value:0/batch_normalization_20/ReadVariableOp_1:value:0>batch_normalization_20/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<¢
%batch_normalization_20/AssignNewValueAssignVariableOp?batch_normalization_20_fusedbatchnormv3_readvariableop_resource4batch_normalization_20/FusedBatchNormV3:batch_mean:07^batch_normalization_20/FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(¬
'batch_normalization_20/AssignNewValue_1AssignVariableOpAbatch_normalization_20_fusedbatchnormv3_readvariableop_1_resource8batch_normalization_20/FusedBatchNormV3:batch_variance:09^batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(`
flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
flatten_6/ReshapeReshape+batch_normalization_20/FusedBatchNormV3:y:0flatten_6/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
dense_12/MatMulMatMulflatten_6/Reshape:output:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
dense_12/ReluReludense_12/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
dense_13/MatMulMatMuldense_12/Relu:activations:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
dense_13/SoftmaxSoftmaxdense_13/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
IdentityIdentitydense_13/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿí

NoOpNoOp&^batch_normalization_18/AssignNewValue(^batch_normalization_18/AssignNewValue_17^batch_normalization_18/FusedBatchNormV3/ReadVariableOp9^batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_18/ReadVariableOp(^batch_normalization_18/ReadVariableOp_1&^batch_normalization_19/AssignNewValue(^batch_normalization_19/AssignNewValue_17^batch_normalization_19/FusedBatchNormV3/ReadVariableOp9^batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_19/ReadVariableOp(^batch_normalization_19/ReadVariableOp_1&^batch_normalization_20/AssignNewValue(^batch_normalization_20/AssignNewValue_17^batch_normalization_20/FusedBatchNormV3/ReadVariableOp9^batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_20/ReadVariableOp(^batch_normalization_20/ReadVariableOp_1!^conv2d_34/BiasAdd/ReadVariableOp ^conv2d_34/Conv2D/ReadVariableOp!^conv2d_35/BiasAdd/ReadVariableOp ^conv2d_35/Conv2D/ReadVariableOp!^conv2d_36/BiasAdd/ReadVariableOp ^conv2d_36/Conv2D/ReadVariableOp!^conv2d_37/BiasAdd/ReadVariableOp ^conv2d_37/Conv2D/ReadVariableOp!^conv2d_38/BiasAdd/ReadVariableOp ^conv2d_38/Conv2D/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:ÿÿÿÿÿÿÿÿÿ8: : : : : : : : : : : : : : : : : : : : : : : : : : 2N
%batch_normalization_18/AssignNewValue%batch_normalization_18/AssignNewValue2R
'batch_normalization_18/AssignNewValue_1'batch_normalization_18/AssignNewValue_12p
6batch_normalization_18/FusedBatchNormV3/ReadVariableOp6batch_normalization_18/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_18/FusedBatchNormV3/ReadVariableOp_18batch_normalization_18/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_18/ReadVariableOp%batch_normalization_18/ReadVariableOp2R
'batch_normalization_18/ReadVariableOp_1'batch_normalization_18/ReadVariableOp_12N
%batch_normalization_19/AssignNewValue%batch_normalization_19/AssignNewValue2R
'batch_normalization_19/AssignNewValue_1'batch_normalization_19/AssignNewValue_12p
6batch_normalization_19/FusedBatchNormV3/ReadVariableOp6batch_normalization_19/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_19/FusedBatchNormV3/ReadVariableOp_18batch_normalization_19/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_19/ReadVariableOp%batch_normalization_19/ReadVariableOp2R
'batch_normalization_19/ReadVariableOp_1'batch_normalization_19/ReadVariableOp_12N
%batch_normalization_20/AssignNewValue%batch_normalization_20/AssignNewValue2R
'batch_normalization_20/AssignNewValue_1'batch_normalization_20/AssignNewValue_12p
6batch_normalization_20/FusedBatchNormV3/ReadVariableOp6batch_normalization_20/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_20/FusedBatchNormV3/ReadVariableOp_18batch_normalization_20/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_20/ReadVariableOp%batch_normalization_20/ReadVariableOp2R
'batch_normalization_20/ReadVariableOp_1'batch_normalization_20/ReadVariableOp_12D
 conv2d_34/BiasAdd/ReadVariableOp conv2d_34/BiasAdd/ReadVariableOp2B
conv2d_34/Conv2D/ReadVariableOpconv2d_34/Conv2D/ReadVariableOp2D
 conv2d_35/BiasAdd/ReadVariableOp conv2d_35/BiasAdd/ReadVariableOp2B
conv2d_35/Conv2D/ReadVariableOpconv2d_35/Conv2D/ReadVariableOp2D
 conv2d_36/BiasAdd/ReadVariableOp conv2d_36/BiasAdd/ReadVariableOp2B
conv2d_36/Conv2D/ReadVariableOpconv2d_36/Conv2D/ReadVariableOp2D
 conv2d_37/BiasAdd/ReadVariableOp conv2d_37/BiasAdd/ReadVariableOp2B
conv2d_37/Conv2D/ReadVariableOpconv2d_37/Conv2D/ReadVariableOp2D
 conv2d_38/BiasAdd/ReadVariableOp conv2d_38/BiasAdd/ReadVariableOp2B
conv2d_38/Conv2D/ReadVariableOpconv2d_38/Conv2D/ReadVariableOp2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8
 
_user_specified_nameinputs


F__inference_conv2d_38_layer_call_and_return_conditional_losses_2432059

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Þ
¢
S__inference_batch_normalization_19_layer_call_and_return_conditional_losses_2431835

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Í
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


F__inference_conv2d_37_layer_call_and_return_conditional_losses_2432032

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿj
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ
: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
Ê
b
F__inference_flatten_6_layer_call_and_return_conditional_losses_2432081

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
õ
¢
+__inference_conv2d_36_layer_call_fn_2433130

inputs"
unknown:@
	unknown_0:	
identity¢StatefulPartitionedCallç
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_36_layer_call_and_return_conditional_losses_2432015x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

ÿ
F__inference_conv2d_35_layer_call_and_return_conditional_losses_2431988

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4@*
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
:ÿÿÿÿÿÿÿÿÿ4@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ6@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ6@
 
_user_specified_nameinputs
Î

*__inference_dense_12_layer_call_fn_2433345

inputs
unknown:

	unknown_0:	
identity¢StatefulPartitionedCallÞ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_12_layer_call_and_return_conditional_losses_2432094p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¥

÷
E__inference_dense_13_layer_call_and_return_conditional_losses_2432111

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ê

*__inference_dense_13_layer_call_fn_2433365

inputs
unknown:	
	unknown_0:
identity¢StatefulPartitionedCallÝ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_13_layer_call_and_return_conditional_losses_2432111o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ê
b
F__inference_flatten_6_layer_call_and_return_conditional_losses_2433336

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿY
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


F__inference_conv2d_38_layer_call_and_return_conditional_losses_2433253

inputs:
conv2d_readvariableop_resource:.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp~
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

i
M__inference_max_pooling2d_18_layer_call_and_return_conditional_losses_2431734

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¬
ß
.__inference_sequential_8_layer_call_fn_2432173
conv2d_34_input!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@
	unknown_6:@$
	unknown_7:@
	unknown_8:	%
	unknown_9:

unknown_10:	

unknown_11:	

unknown_12:	

unknown_13:	

unknown_14:	&

unknown_15:

unknown_16:	

unknown_17:	

unknown_18:	

unknown_19:	

unknown_20:	

unknown_21:


unknown_22:	

unknown_23:	

unknown_24:
identity¢StatefulPartitionedCall±
StatefulPartitionedCallStatefulPartitionedCallconv2d_34_inputunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
:ÿÿÿÿÿÿÿÿÿ*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_sequential_8_layer_call_and_return_conditional_losses_2432118o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:ÿÿÿÿÿÿÿÿÿ8: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:` \
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8
)
_user_specified_nameconv2d_34_input
I
È
I__inference_sequential_8_layer_call_and_return_conditional_losses_2432630
conv2d_34_input+
conv2d_34_2432563:@
conv2d_34_2432565:@+
conv2d_35_2432568:@@
conv2d_35_2432570:@,
batch_normalization_18_2432574:@,
batch_normalization_18_2432576:@,
batch_normalization_18_2432578:@,
batch_normalization_18_2432580:@,
conv2d_36_2432583:@ 
conv2d_36_2432585:	-
conv2d_37_2432588: 
conv2d_37_2432590:	-
batch_normalization_19_2432594:	-
batch_normalization_19_2432596:	-
batch_normalization_19_2432598:	-
batch_normalization_19_2432600:	-
conv2d_38_2432603: 
conv2d_38_2432605:	-
batch_normalization_20_2432609:	-
batch_normalization_20_2432611:	-
batch_normalization_20_2432613:	-
batch_normalization_20_2432615:	$
dense_12_2432619:

dense_12_2432621:	#
dense_13_2432624:	
dense_13_2432626:
identity¢.batch_normalization_18/StatefulPartitionedCall¢.batch_normalization_19/StatefulPartitionedCall¢.batch_normalization_20/StatefulPartitionedCall¢!conv2d_34/StatefulPartitionedCall¢!conv2d_35/StatefulPartitionedCall¢!conv2d_36/StatefulPartitionedCall¢!conv2d_37/StatefulPartitionedCall¢!conv2d_38/StatefulPartitionedCall¢ dense_12/StatefulPartitionedCall¢ dense_13/StatefulPartitionedCall
!conv2d_34/StatefulPartitionedCallStatefulPartitionedCallconv2d_34_inputconv2d_34_2432563conv2d_34_2432565*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ6@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_34_layer_call_and_return_conditional_losses_2431971¦
!conv2d_35/StatefulPartitionedCallStatefulPartitionedCall*conv2d_34/StatefulPartitionedCall:output:0conv2d_35_2432568conv2d_35_2432570*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_35_layer_call_and_return_conditional_losses_2431988ø
 max_pooling2d_18/PartitionedCallPartitionedCall*conv2d_35/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_18_layer_call_and_return_conditional_losses_2431734
.batch_normalization_18/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_18/PartitionedCall:output:0batch_normalization_18_2432574batch_normalization_18_2432576batch_normalization_18_2432578batch_normalization_18_2432580*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_18_layer_call_and_return_conditional_losses_2431790´
!conv2d_36/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_18/StatefulPartitionedCall:output:0conv2d_36_2432583conv2d_36_2432585*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_36_layer_call_and_return_conditional_losses_2432015§
!conv2d_37/StatefulPartitionedCallStatefulPartitionedCall*conv2d_36/StatefulPartitionedCall:output:0conv2d_37_2432588conv2d_37_2432590*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_37_layer_call_and_return_conditional_losses_2432032ù
 max_pooling2d_19/PartitionedCallPartitionedCall*conv2d_37/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_19_layer_call_and_return_conditional_losses_2431810
.batch_normalization_19/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_19/PartitionedCall:output:0batch_normalization_19_2432594batch_normalization_19_2432596batch_normalization_19_2432598batch_normalization_19_2432600*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_19_layer_call_and_return_conditional_losses_2431866´
!conv2d_38/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_19/StatefulPartitionedCall:output:0conv2d_38_2432603conv2d_38_2432605*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_38_layer_call_and_return_conditional_losses_2432059ù
 max_pooling2d_20/PartitionedCallPartitionedCall*conv2d_38/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_20_layer_call_and_return_conditional_losses_2431886
.batch_normalization_20/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_20/PartitionedCall:output:0batch_normalization_20_2432609batch_normalization_20_2432611batch_normalization_20_2432613batch_normalization_20_2432615*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_20_layer_call_and_return_conditional_losses_2431942ð
flatten_6/PartitionedCallPartitionedCall7batch_normalization_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_flatten_6_layer_call_and_return_conditional_losses_2432081
 dense_12/StatefulPartitionedCallStatefulPartitionedCall"flatten_6/PartitionedCall:output:0dense_12_2432619dense_12_2432621*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_12_layer_call_and_return_conditional_losses_2432094
 dense_13/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0dense_13_2432624dense_13_2432626*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_13_layer_call_and_return_conditional_losses_2432111x
IdentityIdentity)dense_13/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÓ
NoOpNoOp/^batch_normalization_18/StatefulPartitionedCall/^batch_normalization_19/StatefulPartitionedCall/^batch_normalization_20/StatefulPartitionedCall"^conv2d_34/StatefulPartitionedCall"^conv2d_35/StatefulPartitionedCall"^conv2d_36/StatefulPartitionedCall"^conv2d_37/StatefulPartitionedCall"^conv2d_38/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:ÿÿÿÿÿÿÿÿÿ8: : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_18/StatefulPartitionedCall.batch_normalization_18/StatefulPartitionedCall2`
.batch_normalization_19/StatefulPartitionedCall.batch_normalization_19/StatefulPartitionedCall2`
.batch_normalization_20/StatefulPartitionedCall.batch_normalization_20/StatefulPartitionedCall2F
!conv2d_34/StatefulPartitionedCall!conv2d_34/StatefulPartitionedCall2F
!conv2d_35/StatefulPartitionedCall!conv2d_35/StatefulPartitionedCall2F
!conv2d_36/StatefulPartitionedCall!conv2d_36/StatefulPartitionedCall2F
!conv2d_37/StatefulPartitionedCall!conv2d_37/StatefulPartitionedCall2F
!conv2d_38/StatefulPartitionedCall!conv2d_38/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall:` \
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8
)
_user_specified_nameconv2d_34_input
ø
£
+__inference_conv2d_37_layer_call_fn_2433150

inputs#
unknown:
	unknown_0:	
identity¢StatefulPartitionedCallç
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_37_layer_call_and_return_conditional_losses_2432032x
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :ÿÿÿÿÿÿÿÿÿ
: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

 
_user_specified_nameinputs
ç®
¨1
#__inference__traced_restore_2433859
file_prefix;
!assignvariableop_conv2d_34_kernel:@/
!assignvariableop_1_conv2d_34_bias:@=
#assignvariableop_2_conv2d_35_kernel:@@/
!assignvariableop_3_conv2d_35_bias:@=
/assignvariableop_4_batch_normalization_18_gamma:@<
.assignvariableop_5_batch_normalization_18_beta:@C
5assignvariableop_6_batch_normalization_18_moving_mean:@G
9assignvariableop_7_batch_normalization_18_moving_variance:@>
#assignvariableop_8_conv2d_36_kernel:@0
!assignvariableop_9_conv2d_36_bias:	@
$assignvariableop_10_conv2d_37_kernel:1
"assignvariableop_11_conv2d_37_bias:	?
0assignvariableop_12_batch_normalization_19_gamma:	>
/assignvariableop_13_batch_normalization_19_beta:	E
6assignvariableop_14_batch_normalization_19_moving_mean:	I
:assignvariableop_15_batch_normalization_19_moving_variance:	@
$assignvariableop_16_conv2d_38_kernel:1
"assignvariableop_17_conv2d_38_bias:	?
0assignvariableop_18_batch_normalization_20_gamma:	>
/assignvariableop_19_batch_normalization_20_beta:	E
6assignvariableop_20_batch_normalization_20_moving_mean:	I
:assignvariableop_21_batch_normalization_20_moving_variance:	7
#assignvariableop_22_dense_12_kernel:
0
!assignvariableop_23_dense_12_bias:	6
#assignvariableop_24_dense_13_kernel:	/
!assignvariableop_25_dense_13_bias:'
assignvariableop_26_adam_iter:	 )
assignvariableop_27_adam_beta_1: )
assignvariableop_28_adam_beta_2: (
assignvariableop_29_adam_decay: 0
&assignvariableop_30_adam_learning_rate: %
assignvariableop_31_total_1: %
assignvariableop_32_count_1: #
assignvariableop_33_total: #
assignvariableop_34_count: E
+assignvariableop_35_adam_conv2d_34_kernel_m:@7
)assignvariableop_36_adam_conv2d_34_bias_m:@E
+assignvariableop_37_adam_conv2d_35_kernel_m:@@7
)assignvariableop_38_adam_conv2d_35_bias_m:@E
7assignvariableop_39_adam_batch_normalization_18_gamma_m:@D
6assignvariableop_40_adam_batch_normalization_18_beta_m:@F
+assignvariableop_41_adam_conv2d_36_kernel_m:@8
)assignvariableop_42_adam_conv2d_36_bias_m:	G
+assignvariableop_43_adam_conv2d_37_kernel_m:8
)assignvariableop_44_adam_conv2d_37_bias_m:	F
7assignvariableop_45_adam_batch_normalization_19_gamma_m:	E
6assignvariableop_46_adam_batch_normalization_19_beta_m:	G
+assignvariableop_47_adam_conv2d_38_kernel_m:8
)assignvariableop_48_adam_conv2d_38_bias_m:	F
7assignvariableop_49_adam_batch_normalization_20_gamma_m:	E
6assignvariableop_50_adam_batch_normalization_20_beta_m:	>
*assignvariableop_51_adam_dense_12_kernel_m:
7
(assignvariableop_52_adam_dense_12_bias_m:	=
*assignvariableop_53_adam_dense_13_kernel_m:	6
(assignvariableop_54_adam_dense_13_bias_m:E
+assignvariableop_55_adam_conv2d_34_kernel_v:@7
)assignvariableop_56_adam_conv2d_34_bias_v:@E
+assignvariableop_57_adam_conv2d_35_kernel_v:@@7
)assignvariableop_58_adam_conv2d_35_bias_v:@E
7assignvariableop_59_adam_batch_normalization_18_gamma_v:@D
6assignvariableop_60_adam_batch_normalization_18_beta_v:@F
+assignvariableop_61_adam_conv2d_36_kernel_v:@8
)assignvariableop_62_adam_conv2d_36_bias_v:	G
+assignvariableop_63_adam_conv2d_37_kernel_v:8
)assignvariableop_64_adam_conv2d_37_bias_v:	F
7assignvariableop_65_adam_batch_normalization_19_gamma_v:	E
6assignvariableop_66_adam_batch_normalization_19_beta_v:	G
+assignvariableop_67_adam_conv2d_38_kernel_v:8
)assignvariableop_68_adam_conv2d_38_bias_v:	F
7assignvariableop_69_adam_batch_normalization_20_gamma_v:	E
6assignvariableop_70_adam_batch_normalization_20_beta_v:	>
*assignvariableop_71_adam_dense_12_kernel_v:
7
(assignvariableop_72_adam_dense_12_bias_v:	=
*assignvariableop_73_adam_dense_13_kernel_v:	6
(assignvariableop_74_adam_dense_13_bias_v:
identity_76¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_55¢AssignVariableOp_56¢AssignVariableOp_57¢AssignVariableOp_58¢AssignVariableOp_59¢AssignVariableOp_6¢AssignVariableOp_60¢AssignVariableOp_61¢AssignVariableOp_62¢AssignVariableOp_63¢AssignVariableOp_64¢AssignVariableOp_65¢AssignVariableOp_66¢AssignVariableOp_67¢AssignVariableOp_68¢AssignVariableOp_69¢AssignVariableOp_7¢AssignVariableOp_70¢AssignVariableOp_71¢AssignVariableOp_72¢AssignVariableOp_73¢AssignVariableOp_74¢AssignVariableOp_8¢AssignVariableOp_9ÿ)
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:L*
dtype0*¥)
value)B)LB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:L*
dtype0*­
value£B LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*Æ
_output_shapes³
°::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*Z
dtypesP
N2L	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_34_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_34_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_35_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_35_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp/assignvariableop_4_batch_normalization_18_gammaIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp.assignvariableop_5_batch_normalization_18_betaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:¤
AssignVariableOp_6AssignVariableOp5assignvariableop_6_batch_normalization_18_moving_meanIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_7AssignVariableOp9assignvariableop_7_batch_normalization_18_moving_varianceIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp#assignvariableop_8_conv2d_36_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp!assignvariableop_9_conv2d_36_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp$assignvariableop_10_conv2d_37_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOp"assignvariableop_11_conv2d_37_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_12AssignVariableOp0assignvariableop_12_batch_normalization_19_gammaIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_13AssignVariableOp/assignvariableop_13_batch_normalization_19_betaIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_14AssignVariableOp6assignvariableop_14_batch_normalization_19_moving_meanIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_15AssignVariableOp:assignvariableop_15_batch_normalization_19_moving_varianceIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp$assignvariableop_16_conv2d_38_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp"assignvariableop_17_conv2d_38_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:¡
AssignVariableOp_18AssignVariableOp0assignvariableop_18_batch_normalization_20_gammaIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_19AssignVariableOp/assignvariableop_19_batch_normalization_20_betaIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_20AssignVariableOp6assignvariableop_20_batch_normalization_20_moving_meanIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:«
AssignVariableOp_21AssignVariableOp:assignvariableop_21_batch_normalization_20_moving_varianceIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp#assignvariableop_22_dense_12_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp!assignvariableop_23_dense_12_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp#assignvariableop_24_dense_13_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_25AssignVariableOp!assignvariableop_25_dense_13_biasIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_26AssignVariableOpassignvariableop_26_adam_iterIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_27AssignVariableOpassignvariableop_27_adam_beta_1Identity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_28AssignVariableOpassignvariableop_28_adam_beta_2Identity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_29AssignVariableOpassignvariableop_29_adam_decayIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_30AssignVariableOp&assignvariableop_30_adam_learning_rateIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_31AssignVariableOpassignvariableop_31_total_1Identity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_32AssignVariableOpassignvariableop_32_count_1Identity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_33AssignVariableOpassignvariableop_33_totalIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_34AssignVariableOpassignvariableop_34_countIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_35AssignVariableOp+assignvariableop_35_adam_conv2d_34_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_36AssignVariableOp)assignvariableop_36_adam_conv2d_34_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_37AssignVariableOp+assignvariableop_37_adam_conv2d_35_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_38AssignVariableOp)assignvariableop_38_adam_conv2d_35_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_39AssignVariableOp7assignvariableop_39_adam_batch_normalization_18_gamma_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_40AssignVariableOp6assignvariableop_40_adam_batch_normalization_18_beta_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_41AssignVariableOp+assignvariableop_41_adam_conv2d_36_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_42AssignVariableOp)assignvariableop_42_adam_conv2d_36_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_43AssignVariableOp+assignvariableop_43_adam_conv2d_37_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_44AssignVariableOp)assignvariableop_44_adam_conv2d_37_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_45AssignVariableOp7assignvariableop_45_adam_batch_normalization_19_gamma_mIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_46AssignVariableOp6assignvariableop_46_adam_batch_normalization_19_beta_mIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_47AssignVariableOp+assignvariableop_47_adam_conv2d_38_kernel_mIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_48AssignVariableOp)assignvariableop_48_adam_conv2d_38_bias_mIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_49AssignVariableOp7assignvariableop_49_adam_batch_normalization_20_gamma_mIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_50AssignVariableOp6assignvariableop_50_adam_batch_normalization_20_beta_mIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_51AssignVariableOp*assignvariableop_51_adam_dense_12_kernel_mIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_52AssignVariableOp(assignvariableop_52_adam_dense_12_bias_mIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_53AssignVariableOp*assignvariableop_53_adam_dense_13_kernel_mIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_54AssignVariableOp(assignvariableop_54_adam_dense_13_bias_mIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_55AssignVariableOp+assignvariableop_55_adam_conv2d_34_kernel_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_56AssignVariableOp)assignvariableop_56_adam_conv2d_34_bias_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_57AssignVariableOp+assignvariableop_57_adam_conv2d_35_kernel_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_58AssignVariableOp)assignvariableop_58_adam_conv2d_35_bias_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_59AssignVariableOp7assignvariableop_59_adam_batch_normalization_18_gamma_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_60AssignVariableOp6assignvariableop_60_adam_batch_normalization_18_beta_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_61AssignVariableOp+assignvariableop_61_adam_conv2d_36_kernel_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_62AssignVariableOp)assignvariableop_62_adam_conv2d_36_bias_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_63AssignVariableOp+assignvariableop_63_adam_conv2d_37_kernel_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_64IdentityRestoreV2:tensors:64"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_64AssignVariableOp)assignvariableop_64_adam_conv2d_37_bias_vIdentity_64:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_65IdentityRestoreV2:tensors:65"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_65AssignVariableOp7assignvariableop_65_adam_batch_normalization_19_gamma_vIdentity_65:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_66IdentityRestoreV2:tensors:66"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_66AssignVariableOp6assignvariableop_66_adam_batch_normalization_19_beta_vIdentity_66:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_67IdentityRestoreV2:tensors:67"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_67AssignVariableOp+assignvariableop_67_adam_conv2d_38_kernel_vIdentity_67:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_68IdentityRestoreV2:tensors:68"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_68AssignVariableOp)assignvariableop_68_adam_conv2d_38_bias_vIdentity_68:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_69IdentityRestoreV2:tensors:69"/device:CPU:0*
T0*
_output_shapes
:¨
AssignVariableOp_69AssignVariableOp7assignvariableop_69_adam_batch_normalization_20_gamma_vIdentity_69:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_70IdentityRestoreV2:tensors:70"/device:CPU:0*
T0*
_output_shapes
:§
AssignVariableOp_70AssignVariableOp6assignvariableop_70_adam_batch_normalization_20_beta_vIdentity_70:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_71IdentityRestoreV2:tensors:71"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_71AssignVariableOp*assignvariableop_71_adam_dense_12_kernel_vIdentity_71:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_72IdentityRestoreV2:tensors:72"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_72AssignVariableOp(assignvariableop_72_adam_dense_12_bias_vIdentity_72:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_73IdentityRestoreV2:tensors:73"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_73AssignVariableOp*assignvariableop_73_adam_dense_13_kernel_vIdentity_73:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_74IdentityRestoreV2:tensors:74"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_74AssignVariableOp(assignvariableop_74_adam_dense_13_bias_vIdentity_74:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 Á
Identity_75Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_76IdentityIdentity_75:output:0^NoOp_1*
T0*
_output_shapes
: ®
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_64^AssignVariableOp_65^AssignVariableOp_66^AssignVariableOp_67^AssignVariableOp_68^AssignVariableOp_69^AssignVariableOp_7^AssignVariableOp_70^AssignVariableOp_71^AssignVariableOp_72^AssignVariableOp_73^AssignVariableOp_74^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_76Identity_76:output:0*­
_input_shapes
: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
	
Ó
8__inference_batch_normalization_18_layer_call_fn_2433072

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_18_layer_call_and_return_conditional_losses_2431759
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
Þ
¢
S__inference_batch_normalization_19_layer_call_and_return_conditional_losses_2433215

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Í
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ó{
Þ
I__inference_sequential_8_layer_call_and_return_conditional_losses_2432909

inputsB
(conv2d_34_conv2d_readvariableop_resource:@7
)conv2d_34_biasadd_readvariableop_resource:@B
(conv2d_35_conv2d_readvariableop_resource:@@7
)conv2d_35_biasadd_readvariableop_resource:@<
.batch_normalization_18_readvariableop_resource:@>
0batch_normalization_18_readvariableop_1_resource:@M
?batch_normalization_18_fusedbatchnormv3_readvariableop_resource:@O
Abatch_normalization_18_fusedbatchnormv3_readvariableop_1_resource:@C
(conv2d_36_conv2d_readvariableop_resource:@8
)conv2d_36_biasadd_readvariableop_resource:	D
(conv2d_37_conv2d_readvariableop_resource:8
)conv2d_37_biasadd_readvariableop_resource:	=
.batch_normalization_19_readvariableop_resource:	?
0batch_normalization_19_readvariableop_1_resource:	N
?batch_normalization_19_fusedbatchnormv3_readvariableop_resource:	P
Abatch_normalization_19_fusedbatchnormv3_readvariableop_1_resource:	D
(conv2d_38_conv2d_readvariableop_resource:8
)conv2d_38_biasadd_readvariableop_resource:	=
.batch_normalization_20_readvariableop_resource:	?
0batch_normalization_20_readvariableop_1_resource:	N
?batch_normalization_20_fusedbatchnormv3_readvariableop_resource:	P
Abatch_normalization_20_fusedbatchnormv3_readvariableop_1_resource:	;
'dense_12_matmul_readvariableop_resource:
7
(dense_12_biasadd_readvariableop_resource:	:
'dense_13_matmul_readvariableop_resource:	6
(dense_13_biasadd_readvariableop_resource:
identity¢6batch_normalization_18/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_18/ReadVariableOp¢'batch_normalization_18/ReadVariableOp_1¢6batch_normalization_19/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_19/ReadVariableOp¢'batch_normalization_19/ReadVariableOp_1¢6batch_normalization_20/FusedBatchNormV3/ReadVariableOp¢8batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1¢%batch_normalization_20/ReadVariableOp¢'batch_normalization_20/ReadVariableOp_1¢ conv2d_34/BiasAdd/ReadVariableOp¢conv2d_34/Conv2D/ReadVariableOp¢ conv2d_35/BiasAdd/ReadVariableOp¢conv2d_35/Conv2D/ReadVariableOp¢ conv2d_36/BiasAdd/ReadVariableOp¢conv2d_36/Conv2D/ReadVariableOp¢ conv2d_37/BiasAdd/ReadVariableOp¢conv2d_37/Conv2D/ReadVariableOp¢ conv2d_38/BiasAdd/ReadVariableOp¢conv2d_38/Conv2D/ReadVariableOp¢dense_12/BiasAdd/ReadVariableOp¢dense_12/MatMul/ReadVariableOp¢dense_13/BiasAdd/ReadVariableOp¢dense_13/MatMul/ReadVariableOp
conv2d_34/Conv2D/ReadVariableOpReadVariableOp(conv2d_34_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0®
conv2d_34/Conv2DConv2Dinputs'conv2d_34/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ6@*
paddingVALID*
strides

 conv2d_34/BiasAdd/ReadVariableOpReadVariableOp)conv2d_34_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_34/BiasAddBiasAddconv2d_34/Conv2D:output:0(conv2d_34/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ6@l
conv2d_34/ReluReluconv2d_34/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ6@
conv2d_35/Conv2D/ReadVariableOpReadVariableOp(conv2d_35_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0Ä
conv2d_35/Conv2DConv2Dconv2d_34/Relu:activations:0'conv2d_35/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4@*
paddingVALID*
strides

 conv2d_35/BiasAdd/ReadVariableOpReadVariableOp)conv2d_35_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
conv2d_35/BiasAddBiasAddconv2d_35/Conv2D:output:0(conv2d_35/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4@l
conv2d_35/ReluReluconv2d_35/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4@®
max_pooling2d_18/MaxPoolMaxPoolconv2d_35/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides

%batch_normalization_18/ReadVariableOpReadVariableOp.batch_normalization_18_readvariableop_resource*
_output_shapes
:@*
dtype0
'batch_normalization_18/ReadVariableOp_1ReadVariableOp0batch_normalization_18_readvariableop_1_resource*
_output_shapes
:@*
dtype0²
6batch_normalization_18/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_18_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0¶
8batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_18_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ä
'batch_normalization_18/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_18/MaxPool:output:0-batch_normalization_18/ReadVariableOp:value:0/batch_normalization_18/ReadVariableOp_1:value:0>batch_normalization_18/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( 
conv2d_36/Conv2D/ReadVariableOpReadVariableOp(conv2d_36_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0Ô
conv2d_36/Conv2DConv2D+batch_normalization_18/FusedBatchNormV3:y:0'conv2d_36/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
paddingVALID*
strides

 conv2d_36/BiasAdd/ReadVariableOpReadVariableOp)conv2d_36_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_36/BiasAddBiasAddconv2d_36/Conv2D:output:0(conv2d_36/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
m
conv2d_36/ReluReluconv2d_36/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

conv2d_37/Conv2D/ReadVariableOpReadVariableOp(conv2d_37_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Å
conv2d_37/Conv2DConv2Dconv2d_36/Relu:activations:0'conv2d_37/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides

 conv2d_37/BiasAdd/ReadVariableOpReadVariableOp)conv2d_37_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_37/BiasAddBiasAddconv2d_37/Conv2D:output:0(conv2d_37/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿm
conv2d_37/ReluReluconv2d_37/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¯
max_pooling2d_19/MaxPoolMaxPoolconv2d_37/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

%batch_normalization_19/ReadVariableOpReadVariableOp.batch_normalization_19_readvariableop_resource*
_output_shapes	
:*
dtype0
'batch_normalization_19/ReadVariableOp_1ReadVariableOp0batch_normalization_19_readvariableop_1_resource*
_output_shapes	
:*
dtype0³
6batch_normalization_19/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_19_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0·
8batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_19_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0É
'batch_normalization_19/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_19/MaxPool:output:0-batch_normalization_19/ReadVariableOp:value:0/batch_normalization_19/ReadVariableOp_1:value:0>batch_normalization_19/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( 
conv2d_38/Conv2D/ReadVariableOpReadVariableOp(conv2d_38_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0Ô
conv2d_38/Conv2DConv2D+batch_normalization_19/FusedBatchNormV3:y:0'conv2d_38/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	*
paddingVALID*
strides

 conv2d_38/BiasAdd/ReadVariableOpReadVariableOp)conv2d_38_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
conv2d_38/BiasAddBiasAddconv2d_38/Conv2D:output:0(conv2d_38/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	m
conv2d_38/ReluReluconv2d_38/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	¯
max_pooling2d_20/MaxPoolMaxPoolconv2d_38/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides

%batch_normalization_20/ReadVariableOpReadVariableOp.batch_normalization_20_readvariableop_resource*
_output_shapes	
:*
dtype0
'batch_normalization_20/ReadVariableOp_1ReadVariableOp0batch_normalization_20_readvariableop_1_resource*
_output_shapes	
:*
dtype0³
6batch_normalization_20/FusedBatchNormV3/ReadVariableOpReadVariableOp?batch_normalization_20_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0·
8batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpAbatch_normalization_20_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0É
'batch_normalization_20/FusedBatchNormV3FusedBatchNormV3!max_pooling2d_20/MaxPool:output:0-batch_normalization_20/ReadVariableOp:value:0/batch_normalization_20/ReadVariableOp_1:value:0>batch_normalization_20/FusedBatchNormV3/ReadVariableOp:value:0@batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( `
flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   
flatten_6/ReshapeReshape+batch_normalization_20/FusedBatchNormV3:y:0flatten_6/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_12/MatMul/ReadVariableOpReadVariableOp'dense_12_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0
dense_12/MatMulMatMulflatten_6/Reshape:output:0&dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_12/BiasAdd/ReadVariableOpReadVariableOp(dense_12_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0
dense_12/BiasAddBiasAdddense_12/MatMul:product:0'dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿc
dense_12/ReluReludense_12/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_13/MatMul/ReadVariableOpReadVariableOp'dense_13_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0
dense_13/MatMulMatMuldense_12/Relu:activations:0&dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
dense_13/BiasAdd/ReadVariableOpReadVariableOp(dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0
dense_13/BiasAddBiasAdddense_13/MatMul:product:0'dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿh
dense_13/SoftmaxSoftmaxdense_13/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿi
IdentityIdentitydense_13/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ÷
NoOpNoOp7^batch_normalization_18/FusedBatchNormV3/ReadVariableOp9^batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_18/ReadVariableOp(^batch_normalization_18/ReadVariableOp_17^batch_normalization_19/FusedBatchNormV3/ReadVariableOp9^batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_19/ReadVariableOp(^batch_normalization_19/ReadVariableOp_17^batch_normalization_20/FusedBatchNormV3/ReadVariableOp9^batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1&^batch_normalization_20/ReadVariableOp(^batch_normalization_20/ReadVariableOp_1!^conv2d_34/BiasAdd/ReadVariableOp ^conv2d_34/Conv2D/ReadVariableOp!^conv2d_35/BiasAdd/ReadVariableOp ^conv2d_35/Conv2D/ReadVariableOp!^conv2d_36/BiasAdd/ReadVariableOp ^conv2d_36/Conv2D/ReadVariableOp!^conv2d_37/BiasAdd/ReadVariableOp ^conv2d_37/Conv2D/ReadVariableOp!^conv2d_38/BiasAdd/ReadVariableOp ^conv2d_38/Conv2D/ReadVariableOp ^dense_12/BiasAdd/ReadVariableOp^dense_12/MatMul/ReadVariableOp ^dense_13/BiasAdd/ReadVariableOp^dense_13/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:ÿÿÿÿÿÿÿÿÿ8: : : : : : : : : : : : : : : : : : : : : : : : : : 2p
6batch_normalization_18/FusedBatchNormV3/ReadVariableOp6batch_normalization_18/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_18/FusedBatchNormV3/ReadVariableOp_18batch_normalization_18/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_18/ReadVariableOp%batch_normalization_18/ReadVariableOp2R
'batch_normalization_18/ReadVariableOp_1'batch_normalization_18/ReadVariableOp_12p
6batch_normalization_19/FusedBatchNormV3/ReadVariableOp6batch_normalization_19/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_19/FusedBatchNormV3/ReadVariableOp_18batch_normalization_19/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_19/ReadVariableOp%batch_normalization_19/ReadVariableOp2R
'batch_normalization_19/ReadVariableOp_1'batch_normalization_19/ReadVariableOp_12p
6batch_normalization_20/FusedBatchNormV3/ReadVariableOp6batch_normalization_20/FusedBatchNormV3/ReadVariableOp2t
8batch_normalization_20/FusedBatchNormV3/ReadVariableOp_18batch_normalization_20/FusedBatchNormV3/ReadVariableOp_12N
%batch_normalization_20/ReadVariableOp%batch_normalization_20/ReadVariableOp2R
'batch_normalization_20/ReadVariableOp_1'batch_normalization_20/ReadVariableOp_12D
 conv2d_34/BiasAdd/ReadVariableOp conv2d_34/BiasAdd/ReadVariableOp2B
conv2d_34/Conv2D/ReadVariableOpconv2d_34/Conv2D/ReadVariableOp2D
 conv2d_35/BiasAdd/ReadVariableOp conv2d_35/BiasAdd/ReadVariableOp2B
conv2d_35/Conv2D/ReadVariableOpconv2d_35/Conv2D/ReadVariableOp2D
 conv2d_36/BiasAdd/ReadVariableOp conv2d_36/BiasAdd/ReadVariableOp2B
conv2d_36/Conv2D/ReadVariableOpconv2d_36/Conv2D/ReadVariableOp2D
 conv2d_37/BiasAdd/ReadVariableOp conv2d_37/BiasAdd/ReadVariableOp2B
conv2d_37/Conv2D/ReadVariableOpconv2d_37/Conv2D/ReadVariableOp2D
 conv2d_38/BiasAdd/ReadVariableOp conv2d_38/BiasAdd/ReadVariableOp2B
conv2d_38/Conv2D/ReadVariableOpconv2d_38/Conv2D/ReadVariableOp2B
dense_12/BiasAdd/ReadVariableOpdense_12/BiasAdd/ReadVariableOp2@
dense_12/MatMul/ReadVariableOpdense_12/MatMul/ReadVariableOp2B
dense_13/BiasAdd/ReadVariableOpdense_13/BiasAdd/ReadVariableOp2@
dense_13/MatMul/ReadVariableOpdense_13/MatMul/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8
 
_user_specified_nameinputs
õ
ä
"__inference__wrapped_model_2431725
conv2d_34_inputO
5sequential_8_conv2d_34_conv2d_readvariableop_resource:@D
6sequential_8_conv2d_34_biasadd_readvariableop_resource:@O
5sequential_8_conv2d_35_conv2d_readvariableop_resource:@@D
6sequential_8_conv2d_35_biasadd_readvariableop_resource:@I
;sequential_8_batch_normalization_18_readvariableop_resource:@K
=sequential_8_batch_normalization_18_readvariableop_1_resource:@Z
Lsequential_8_batch_normalization_18_fusedbatchnormv3_readvariableop_resource:@\
Nsequential_8_batch_normalization_18_fusedbatchnormv3_readvariableop_1_resource:@P
5sequential_8_conv2d_36_conv2d_readvariableop_resource:@E
6sequential_8_conv2d_36_biasadd_readvariableop_resource:	Q
5sequential_8_conv2d_37_conv2d_readvariableop_resource:E
6sequential_8_conv2d_37_biasadd_readvariableop_resource:	J
;sequential_8_batch_normalization_19_readvariableop_resource:	L
=sequential_8_batch_normalization_19_readvariableop_1_resource:	[
Lsequential_8_batch_normalization_19_fusedbatchnormv3_readvariableop_resource:	]
Nsequential_8_batch_normalization_19_fusedbatchnormv3_readvariableop_1_resource:	Q
5sequential_8_conv2d_38_conv2d_readvariableop_resource:E
6sequential_8_conv2d_38_biasadd_readvariableop_resource:	J
;sequential_8_batch_normalization_20_readvariableop_resource:	L
=sequential_8_batch_normalization_20_readvariableop_1_resource:	[
Lsequential_8_batch_normalization_20_fusedbatchnormv3_readvariableop_resource:	]
Nsequential_8_batch_normalization_20_fusedbatchnormv3_readvariableop_1_resource:	H
4sequential_8_dense_12_matmul_readvariableop_resource:
D
5sequential_8_dense_12_biasadd_readvariableop_resource:	G
4sequential_8_dense_13_matmul_readvariableop_resource:	C
5sequential_8_dense_13_biasadd_readvariableop_resource:
identity¢Csequential_8/batch_normalization_18/FusedBatchNormV3/ReadVariableOp¢Esequential_8/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1¢2sequential_8/batch_normalization_18/ReadVariableOp¢4sequential_8/batch_normalization_18/ReadVariableOp_1¢Csequential_8/batch_normalization_19/FusedBatchNormV3/ReadVariableOp¢Esequential_8/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1¢2sequential_8/batch_normalization_19/ReadVariableOp¢4sequential_8/batch_normalization_19/ReadVariableOp_1¢Csequential_8/batch_normalization_20/FusedBatchNormV3/ReadVariableOp¢Esequential_8/batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1¢2sequential_8/batch_normalization_20/ReadVariableOp¢4sequential_8/batch_normalization_20/ReadVariableOp_1¢-sequential_8/conv2d_34/BiasAdd/ReadVariableOp¢,sequential_8/conv2d_34/Conv2D/ReadVariableOp¢-sequential_8/conv2d_35/BiasAdd/ReadVariableOp¢,sequential_8/conv2d_35/Conv2D/ReadVariableOp¢-sequential_8/conv2d_36/BiasAdd/ReadVariableOp¢,sequential_8/conv2d_36/Conv2D/ReadVariableOp¢-sequential_8/conv2d_37/BiasAdd/ReadVariableOp¢,sequential_8/conv2d_37/Conv2D/ReadVariableOp¢-sequential_8/conv2d_38/BiasAdd/ReadVariableOp¢,sequential_8/conv2d_38/Conv2D/ReadVariableOp¢,sequential_8/dense_12/BiasAdd/ReadVariableOp¢+sequential_8/dense_12/MatMul/ReadVariableOp¢,sequential_8/dense_13/BiasAdd/ReadVariableOp¢+sequential_8/dense_13/MatMul/ReadVariableOpª
,sequential_8/conv2d_34/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_34_conv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0Ñ
sequential_8/conv2d_34/Conv2DConv2Dconv2d_34_input4sequential_8/conv2d_34/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ6@*
paddingVALID*
strides
 
-sequential_8/conv2d_34/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_34_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Â
sequential_8/conv2d_34/BiasAddBiasAdd&sequential_8/conv2d_34/Conv2D:output:05sequential_8/conv2d_34/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ6@
sequential_8/conv2d_34/ReluRelu'sequential_8/conv2d_34/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ6@ª
,sequential_8/conv2d_35/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_35_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0ë
sequential_8/conv2d_35/Conv2DConv2D)sequential_8/conv2d_34/Relu:activations:04sequential_8/conv2d_35/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4@*
paddingVALID*
strides
 
-sequential_8/conv2d_35/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_35_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Â
sequential_8/conv2d_35/BiasAddBiasAdd&sequential_8/conv2d_35/Conv2D:output:05sequential_8/conv2d_35/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4@
sequential_8/conv2d_35/ReluRelu'sequential_8/conv2d_35/BiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4@È
%sequential_8/max_pooling2d_18/MaxPoolMaxPool)sequential_8/conv2d_35/Relu:activations:0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*
ksize
*
paddingVALID*
strides
ª
2sequential_8/batch_normalization_18/ReadVariableOpReadVariableOp;sequential_8_batch_normalization_18_readvariableop_resource*
_output_shapes
:@*
dtype0®
4sequential_8/batch_normalization_18/ReadVariableOp_1ReadVariableOp=sequential_8_batch_normalization_18_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ì
Csequential_8/batch_normalization_18/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_8_batch_normalization_18_fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0Ð
Esequential_8/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_8_batch_normalization_18_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0
4sequential_8/batch_normalization_18/FusedBatchNormV3FusedBatchNormV3.sequential_8/max_pooling2d_18/MaxPool:output:0:sequential_8/batch_normalization_18/ReadVariableOp:value:0<sequential_8/batch_normalization_18/ReadVariableOp_1:value:0Ksequential_8/batch_normalization_18/FusedBatchNormV3/ReadVariableOp:value:0Msequential_8/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*K
_output_shapes9
7:ÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( «
,sequential_8/conv2d_36/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_36_conv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0û
sequential_8/conv2d_36/Conv2DConv2D8sequential_8/batch_normalization_18/FusedBatchNormV3:y:04sequential_8/conv2d_36/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
paddingVALID*
strides
¡
-sequential_8/conv2d_36/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_36_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ã
sequential_8/conv2d_36/BiasAddBiasAdd&sequential_8/conv2d_36/Conv2D:output:05sequential_8/conv2d_36/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ

sequential_8/conv2d_36/ReluRelu'sequential_8/conv2d_36/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
¬
,sequential_8/conv2d_37/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_37_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0ì
sequential_8/conv2d_37/Conv2DConv2D)sequential_8/conv2d_36/Relu:activations:04sequential_8/conv2d_37/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
paddingVALID*
strides
¡
-sequential_8/conv2d_37/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_37_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ã
sequential_8/conv2d_37/BiasAddBiasAdd&sequential_8/conv2d_37/Conv2D:output:05sequential_8/conv2d_37/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential_8/conv2d_37/ReluRelu'sequential_8/conv2d_37/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿÉ
%sequential_8/max_pooling2d_19/MaxPoolMaxPool)sequential_8/conv2d_37/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
«
2sequential_8/batch_normalization_19/ReadVariableOpReadVariableOp;sequential_8_batch_normalization_19_readvariableop_resource*
_output_shapes	
:*
dtype0¯
4sequential_8/batch_normalization_19/ReadVariableOp_1ReadVariableOp=sequential_8_batch_normalization_19_readvariableop_1_resource*
_output_shapes	
:*
dtype0Í
Csequential_8/batch_normalization_19/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_8_batch_normalization_19_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0Ñ
Esequential_8/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_8_batch_normalization_19_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0
4sequential_8/batch_normalization_19/FusedBatchNormV3FusedBatchNormV3.sequential_8/max_pooling2d_19/MaxPool:output:0:sequential_8/batch_normalization_19/ReadVariableOp:value:0<sequential_8/batch_normalization_19/ReadVariableOp_1:value:0Ksequential_8/batch_normalization_19/FusedBatchNormV3/ReadVariableOp:value:0Msequential_8/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( ¬
,sequential_8/conv2d_38/Conv2D/ReadVariableOpReadVariableOp5sequential_8_conv2d_38_conv2d_readvariableop_resource*(
_output_shapes
:*
dtype0û
sequential_8/conv2d_38/Conv2DConv2D8sequential_8/batch_normalization_19/FusedBatchNormV3:y:04sequential_8/conv2d_38/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	*
paddingVALID*
strides
¡
-sequential_8/conv2d_38/BiasAdd/ReadVariableOpReadVariableOp6sequential_8_conv2d_38_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0Ã
sequential_8/conv2d_38/BiasAddBiasAdd&sequential_8/conv2d_38/Conv2D:output:05sequential_8/conv2d_38/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	
sequential_8/conv2d_38/ReluRelu'sequential_8/conv2d_38/BiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	É
%sequential_8/max_pooling2d_20/MaxPoolMaxPool)sequential_8/conv2d_38/Relu:activations:0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
«
2sequential_8/batch_normalization_20/ReadVariableOpReadVariableOp;sequential_8_batch_normalization_20_readvariableop_resource*
_output_shapes	
:*
dtype0¯
4sequential_8/batch_normalization_20/ReadVariableOp_1ReadVariableOp=sequential_8_batch_normalization_20_readvariableop_1_resource*
_output_shapes	
:*
dtype0Í
Csequential_8/batch_normalization_20/FusedBatchNormV3/ReadVariableOpReadVariableOpLsequential_8_batch_normalization_20_fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0Ñ
Esequential_8/batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1ReadVariableOpNsequential_8_batch_normalization_20_fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0
4sequential_8/batch_normalization_20/FusedBatchNormV3FusedBatchNormV3.sequential_8/max_pooling2d_20/MaxPool:output:0:sequential_8/batch_normalization_20/ReadVariableOp:value:0<sequential_8/batch_normalization_20/ReadVariableOp_1:value:0Ksequential_8/batch_normalization_20/FusedBatchNormV3/ReadVariableOp:value:0Msequential_8/batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*P
_output_shapes>
<:ÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( m
sequential_8/flatten_6/ConstConst*
_output_shapes
:*
dtype0*
valueB"ÿÿÿÿ   ½
sequential_8/flatten_6/ReshapeReshape8sequential_8/batch_normalization_20/FusedBatchNormV3:y:0%sequential_8/flatten_6/Const:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¢
+sequential_8/dense_12/MatMul/ReadVariableOpReadVariableOp4sequential_8_dense_12_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype0·
sequential_8/dense_12/MatMulMatMul'sequential_8/flatten_6/Reshape:output:03sequential_8/dense_12/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,sequential_8/dense_12/BiasAdd/ReadVariableOpReadVariableOp5sequential_8_dense_12_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype0¹
sequential_8/dense_12/BiasAddBiasAdd&sequential_8/dense_12/MatMul:product:04sequential_8/dense_12/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ}
sequential_8/dense_12/ReluRelu&sequential_8/dense_12/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ¡
+sequential_8/dense_13/MatMul/ReadVariableOpReadVariableOp4sequential_8_dense_13_matmul_readvariableop_resource*
_output_shapes
:	*
dtype0·
sequential_8/dense_13/MatMulMatMul(sequential_8/dense_12/Relu:activations:03sequential_8/dense_13/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
,sequential_8/dense_13/BiasAdd/ReadVariableOpReadVariableOp5sequential_8_dense_13_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0¸
sequential_8/dense_13/BiasAddBiasAdd&sequential_8/dense_13/MatMul:product:04sequential_8/dense_13/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
sequential_8/dense_13/SoftmaxSoftmax&sequential_8/dense_13/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿv
IdentityIdentity'sequential_8/dense_13/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÉ
NoOpNoOpD^sequential_8/batch_normalization_18/FusedBatchNormV3/ReadVariableOpF^sequential_8/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_13^sequential_8/batch_normalization_18/ReadVariableOp5^sequential_8/batch_normalization_18/ReadVariableOp_1D^sequential_8/batch_normalization_19/FusedBatchNormV3/ReadVariableOpF^sequential_8/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_13^sequential_8/batch_normalization_19/ReadVariableOp5^sequential_8/batch_normalization_19/ReadVariableOp_1D^sequential_8/batch_normalization_20/FusedBatchNormV3/ReadVariableOpF^sequential_8/batch_normalization_20/FusedBatchNormV3/ReadVariableOp_13^sequential_8/batch_normalization_20/ReadVariableOp5^sequential_8/batch_normalization_20/ReadVariableOp_1.^sequential_8/conv2d_34/BiasAdd/ReadVariableOp-^sequential_8/conv2d_34/Conv2D/ReadVariableOp.^sequential_8/conv2d_35/BiasAdd/ReadVariableOp-^sequential_8/conv2d_35/Conv2D/ReadVariableOp.^sequential_8/conv2d_36/BiasAdd/ReadVariableOp-^sequential_8/conv2d_36/Conv2D/ReadVariableOp.^sequential_8/conv2d_37/BiasAdd/ReadVariableOp-^sequential_8/conv2d_37/Conv2D/ReadVariableOp.^sequential_8/conv2d_38/BiasAdd/ReadVariableOp-^sequential_8/conv2d_38/Conv2D/ReadVariableOp-^sequential_8/dense_12/BiasAdd/ReadVariableOp,^sequential_8/dense_12/MatMul/ReadVariableOp-^sequential_8/dense_13/BiasAdd/ReadVariableOp,^sequential_8/dense_13/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:ÿÿÿÿÿÿÿÿÿ8: : : : : : : : : : : : : : : : : : : : : : : : : : 2
Csequential_8/batch_normalization_18/FusedBatchNormV3/ReadVariableOpCsequential_8/batch_normalization_18/FusedBatchNormV3/ReadVariableOp2
Esequential_8/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_1Esequential_8/batch_normalization_18/FusedBatchNormV3/ReadVariableOp_12h
2sequential_8/batch_normalization_18/ReadVariableOp2sequential_8/batch_normalization_18/ReadVariableOp2l
4sequential_8/batch_normalization_18/ReadVariableOp_14sequential_8/batch_normalization_18/ReadVariableOp_12
Csequential_8/batch_normalization_19/FusedBatchNormV3/ReadVariableOpCsequential_8/batch_normalization_19/FusedBatchNormV3/ReadVariableOp2
Esequential_8/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_1Esequential_8/batch_normalization_19/FusedBatchNormV3/ReadVariableOp_12h
2sequential_8/batch_normalization_19/ReadVariableOp2sequential_8/batch_normalization_19/ReadVariableOp2l
4sequential_8/batch_normalization_19/ReadVariableOp_14sequential_8/batch_normalization_19/ReadVariableOp_12
Csequential_8/batch_normalization_20/FusedBatchNormV3/ReadVariableOpCsequential_8/batch_normalization_20/FusedBatchNormV3/ReadVariableOp2
Esequential_8/batch_normalization_20/FusedBatchNormV3/ReadVariableOp_1Esequential_8/batch_normalization_20/FusedBatchNormV3/ReadVariableOp_12h
2sequential_8/batch_normalization_20/ReadVariableOp2sequential_8/batch_normalization_20/ReadVariableOp2l
4sequential_8/batch_normalization_20/ReadVariableOp_14sequential_8/batch_normalization_20/ReadVariableOp_12^
-sequential_8/conv2d_34/BiasAdd/ReadVariableOp-sequential_8/conv2d_34/BiasAdd/ReadVariableOp2\
,sequential_8/conv2d_34/Conv2D/ReadVariableOp,sequential_8/conv2d_34/Conv2D/ReadVariableOp2^
-sequential_8/conv2d_35/BiasAdd/ReadVariableOp-sequential_8/conv2d_35/BiasAdd/ReadVariableOp2\
,sequential_8/conv2d_35/Conv2D/ReadVariableOp,sequential_8/conv2d_35/Conv2D/ReadVariableOp2^
-sequential_8/conv2d_36/BiasAdd/ReadVariableOp-sequential_8/conv2d_36/BiasAdd/ReadVariableOp2\
,sequential_8/conv2d_36/Conv2D/ReadVariableOp,sequential_8/conv2d_36/Conv2D/ReadVariableOp2^
-sequential_8/conv2d_37/BiasAdd/ReadVariableOp-sequential_8/conv2d_37/BiasAdd/ReadVariableOp2\
,sequential_8/conv2d_37/Conv2D/ReadVariableOp,sequential_8/conv2d_37/Conv2D/ReadVariableOp2^
-sequential_8/conv2d_38/BiasAdd/ReadVariableOp-sequential_8/conv2d_38/BiasAdd/ReadVariableOp2\
,sequential_8/conv2d_38/Conv2D/ReadVariableOp,sequential_8/conv2d_38/Conv2D/ReadVariableOp2\
,sequential_8/dense_12/BiasAdd/ReadVariableOp,sequential_8/dense_12/BiasAdd/ReadVariableOp2Z
+sequential_8/dense_12/MatMul/ReadVariableOp+sequential_8/dense_12/MatMul/ReadVariableOp2\
,sequential_8/dense_13/BiasAdd/ReadVariableOp,sequential_8/dense_13/BiasAdd/ReadVariableOp2Z
+sequential_8/dense_13/MatMul/ReadVariableOp+sequential_8/dense_13/MatMul/ReadVariableOp:` \
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8
)
_user_specified_nameconv2d_34_input
Þ
¢
S__inference_batch_normalization_20_layer_call_and_return_conditional_losses_2431911

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Í
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
is_training( ~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

i
M__inference_max_pooling2d_19_layer_call_and_return_conditional_losses_2433171

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
	
×
8__inference_batch_normalization_20_layer_call_fn_2433289

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_20_layer_call_and_return_conditional_losses_2431942
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Ö
.__inference_sequential_8_layer_call_fn_2432752

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@
	unknown_6:@$
	unknown_7:@
	unknown_8:	%
	unknown_9:

unknown_10:	

unknown_11:	

unknown_12:	

unknown_13:	

unknown_14:	&

unknown_15:

unknown_16:	

unknown_17:	

unknown_18:	

unknown_19:	

unknown_20:	

unknown_21:


unknown_22:	

unknown_23:	

unknown_24:
identity¢StatefulPartitionedCall¨
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
:ÿÿÿÿÿÿÿÿÿ*<
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_sequential_8_layer_call_and_return_conditional_losses_2432118o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:ÿÿÿÿÿÿÿÿÿ8: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8
 
_user_specified_nameinputs
òH
¿
I__inference_sequential_8_layer_call_and_return_conditional_losses_2432118

inputs+
conv2d_34_2431972:@
conv2d_34_2431974:@+
conv2d_35_2431989:@@
conv2d_35_2431991:@,
batch_normalization_18_2431995:@,
batch_normalization_18_2431997:@,
batch_normalization_18_2431999:@,
batch_normalization_18_2432001:@,
conv2d_36_2432016:@ 
conv2d_36_2432018:	-
conv2d_37_2432033: 
conv2d_37_2432035:	-
batch_normalization_19_2432039:	-
batch_normalization_19_2432041:	-
batch_normalization_19_2432043:	-
batch_normalization_19_2432045:	-
conv2d_38_2432060: 
conv2d_38_2432062:	-
batch_normalization_20_2432066:	-
batch_normalization_20_2432068:	-
batch_normalization_20_2432070:	-
batch_normalization_20_2432072:	$
dense_12_2432095:

dense_12_2432097:	#
dense_13_2432112:	
dense_13_2432114:
identity¢.batch_normalization_18/StatefulPartitionedCall¢.batch_normalization_19/StatefulPartitionedCall¢.batch_normalization_20/StatefulPartitionedCall¢!conv2d_34/StatefulPartitionedCall¢!conv2d_35/StatefulPartitionedCall¢!conv2d_36/StatefulPartitionedCall¢!conv2d_37/StatefulPartitionedCall¢!conv2d_38/StatefulPartitionedCall¢ dense_12/StatefulPartitionedCall¢ dense_13/StatefulPartitionedCall
!conv2d_34/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_34_2431972conv2d_34_2431974*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ6@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_34_layer_call_and_return_conditional_losses_2431971¦
!conv2d_35/StatefulPartitionedCallStatefulPartitionedCall*conv2d_34/StatefulPartitionedCall:output:0conv2d_35_2431989conv2d_35_2431991*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_35_layer_call_and_return_conditional_losses_2431988ø
 max_pooling2d_18/PartitionedCallPartitionedCall*conv2d_35/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_18_layer_call_and_return_conditional_losses_2431734
.batch_normalization_18/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_18/PartitionedCall:output:0batch_normalization_18_2431995batch_normalization_18_2431997batch_normalization_18_2431999batch_normalization_18_2432001*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_18_layer_call_and_return_conditional_losses_2431759´
!conv2d_36/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_18/StatefulPartitionedCall:output:0conv2d_36_2432016conv2d_36_2432018*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_36_layer_call_and_return_conditional_losses_2432015§
!conv2d_37/StatefulPartitionedCallStatefulPartitionedCall*conv2d_36/StatefulPartitionedCall:output:0conv2d_37_2432033conv2d_37_2432035*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_37_layer_call_and_return_conditional_losses_2432032ù
 max_pooling2d_19/PartitionedCallPartitionedCall*conv2d_37/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_19_layer_call_and_return_conditional_losses_2431810
.batch_normalization_19/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_19/PartitionedCall:output:0batch_normalization_19_2432039batch_normalization_19_2432041batch_normalization_19_2432043batch_normalization_19_2432045*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_19_layer_call_and_return_conditional_losses_2431835´
!conv2d_38/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_19/StatefulPartitionedCall:output:0conv2d_38_2432060conv2d_38_2432062*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_38_layer_call_and_return_conditional_losses_2432059ù
 max_pooling2d_20/PartitionedCallPartitionedCall*conv2d_38/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_20_layer_call_and_return_conditional_losses_2431886
.batch_normalization_20/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_20/PartitionedCall:output:0batch_normalization_20_2432066batch_normalization_20_2432068batch_normalization_20_2432070batch_normalization_20_2432072*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_20_layer_call_and_return_conditional_losses_2431911ð
flatten_6/PartitionedCallPartitionedCall7batch_normalization_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_flatten_6_layer_call_and_return_conditional_losses_2432081
 dense_12/StatefulPartitionedCallStatefulPartitionedCall"flatten_6/PartitionedCall:output:0dense_12_2432095dense_12_2432097*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_12_layer_call_and_return_conditional_losses_2432094
 dense_13/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0dense_13_2432112dense_13_2432114*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_13_layer_call_and_return_conditional_losses_2432111x
IdentityIdentity)dense_13/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÓ
NoOpNoOp/^batch_normalization_18/StatefulPartitionedCall/^batch_normalization_19/StatefulPartitionedCall/^batch_normalization_20/StatefulPartitionedCall"^conv2d_34/StatefulPartitionedCall"^conv2d_35/StatefulPartitionedCall"^conv2d_36/StatefulPartitionedCall"^conv2d_37/StatefulPartitionedCall"^conv2d_38/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:ÿÿÿÿÿÿÿÿÿ8: : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_18/StatefulPartitionedCall.batch_normalization_18/StatefulPartitionedCall2`
.batch_normalization_19/StatefulPartitionedCall.batch_normalization_19/StatefulPartitionedCall2`
.batch_normalization_20/StatefulPartitionedCall.batch_normalization_20/StatefulPartitionedCall2F
!conv2d_34/StatefulPartitionedCall!conv2d_34/StatefulPartitionedCall2F
!conv2d_35/StatefulPartitionedCall!conv2d_35/StatefulPartitionedCall2F
!conv2d_36/StatefulPartitionedCall!conv2d_36/StatefulPartitionedCall2F
!conv2d_37/StatefulPartitionedCall!conv2d_37/StatefulPartitionedCall2F
!conv2d_38/StatefulPartitionedCall!conv2d_38/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8
 
_user_specified_nameinputs

Æ
S__inference_batch_normalization_19_layer_call_and_return_conditional_losses_2433233

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Û
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<Æ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ð
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÔ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Â
S__inference_batch_normalization_18_layer_call_and_return_conditional_losses_2431790

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<Æ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ð
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@Ô
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
 	
×
8__inference_batch_normalization_19_layer_call_fn_2433184

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_19_layer_call_and_return_conditional_losses_2431835
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

i
M__inference_max_pooling2d_20_layer_call_and_return_conditional_losses_2431886

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¨

ù
E__inference_dense_12_layer_call_and_return_conditional_losses_2432094

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ÿ
F__inference_conv2d_34_layer_call_and_return_conditional_losses_2433029

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ6@*
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
:ÿÿÿÿÿÿÿÿÿ6@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ6@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ6@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ8: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8
 
_user_specified_nameinputs
	
Ó
8__inference_batch_normalization_18_layer_call_fn_2433085

inputs
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_18_layer_call_and_return_conditional_losses_2431790
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
 	
×
8__inference_batch_normalization_20_layer_call_fn_2433276

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_20_layer_call_and_return_conditional_losses_2431911
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Î

S__inference_batch_normalization_18_layer_call_and_return_conditional_losses_2433103

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
	
×
8__inference_batch_normalization_19_layer_call_fn_2433197

inputs
unknown:	
	unknown_0:	
	unknown_1:	
	unknown_2:	
identity¢StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_19_layer_call_and_return_conditional_losses_2431866
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¨

ù
E__inference_dense_12_layer_call_and_return_conditional_losses_2433356

inputs2
matmul_readvariableop_resource:
.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpv
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿs
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿQ
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿb
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¸
G
+__inference_flatten_6_layer_call_fn_2433330

inputs
identityµ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_flatten_6_layer_call_and_return_conditional_losses_2432081a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:X T
0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¿
´!
 __inference__traced_save_2433624
file_prefix/
+savev2_conv2d_34_kernel_read_readvariableop-
)savev2_conv2d_34_bias_read_readvariableop/
+savev2_conv2d_35_kernel_read_readvariableop-
)savev2_conv2d_35_bias_read_readvariableop;
7savev2_batch_normalization_18_gamma_read_readvariableop:
6savev2_batch_normalization_18_beta_read_readvariableopA
=savev2_batch_normalization_18_moving_mean_read_readvariableopE
Asavev2_batch_normalization_18_moving_variance_read_readvariableop/
+savev2_conv2d_36_kernel_read_readvariableop-
)savev2_conv2d_36_bias_read_readvariableop/
+savev2_conv2d_37_kernel_read_readvariableop-
)savev2_conv2d_37_bias_read_readvariableop;
7savev2_batch_normalization_19_gamma_read_readvariableop:
6savev2_batch_normalization_19_beta_read_readvariableopA
=savev2_batch_normalization_19_moving_mean_read_readvariableopE
Asavev2_batch_normalization_19_moving_variance_read_readvariableop/
+savev2_conv2d_38_kernel_read_readvariableop-
)savev2_conv2d_38_bias_read_readvariableop;
7savev2_batch_normalization_20_gamma_read_readvariableop:
6savev2_batch_normalization_20_beta_read_readvariableopA
=savev2_batch_normalization_20_moving_mean_read_readvariableopE
Asavev2_batch_normalization_20_moving_variance_read_readvariableop.
*savev2_dense_12_kernel_read_readvariableop,
(savev2_dense_12_bias_read_readvariableop.
*savev2_dense_13_kernel_read_readvariableop,
(savev2_dense_13_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop6
2savev2_adam_conv2d_34_kernel_m_read_readvariableop4
0savev2_adam_conv2d_34_bias_m_read_readvariableop6
2savev2_adam_conv2d_35_kernel_m_read_readvariableop4
0savev2_adam_conv2d_35_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_18_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_18_beta_m_read_readvariableop6
2savev2_adam_conv2d_36_kernel_m_read_readvariableop4
0savev2_adam_conv2d_36_bias_m_read_readvariableop6
2savev2_adam_conv2d_37_kernel_m_read_readvariableop4
0savev2_adam_conv2d_37_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_19_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_19_beta_m_read_readvariableop6
2savev2_adam_conv2d_38_kernel_m_read_readvariableop4
0savev2_adam_conv2d_38_bias_m_read_readvariableopB
>savev2_adam_batch_normalization_20_gamma_m_read_readvariableopA
=savev2_adam_batch_normalization_20_beta_m_read_readvariableop5
1savev2_adam_dense_12_kernel_m_read_readvariableop3
/savev2_adam_dense_12_bias_m_read_readvariableop5
1savev2_adam_dense_13_kernel_m_read_readvariableop3
/savev2_adam_dense_13_bias_m_read_readvariableop6
2savev2_adam_conv2d_34_kernel_v_read_readvariableop4
0savev2_adam_conv2d_34_bias_v_read_readvariableop6
2savev2_adam_conv2d_35_kernel_v_read_readvariableop4
0savev2_adam_conv2d_35_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_18_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_18_beta_v_read_readvariableop6
2savev2_adam_conv2d_36_kernel_v_read_readvariableop4
0savev2_adam_conv2d_36_bias_v_read_readvariableop6
2savev2_adam_conv2d_37_kernel_v_read_readvariableop4
0savev2_adam_conv2d_37_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_19_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_19_beta_v_read_readvariableop6
2savev2_adam_conv2d_38_kernel_v_read_readvariableop4
0savev2_adam_conv2d_38_bias_v_read_readvariableopB
>savev2_adam_batch_normalization_20_gamma_v_read_readvariableopA
=savev2_adam_batch_normalization_20_beta_v_read_readvariableop5
1savev2_adam_dense_12_kernel_v_read_readvariableop3
/savev2_adam_dense_12_bias_v_read_readvariableop5
1savev2_adam_dense_13_kernel_v_read_readvariableop3
/savev2_adam_dense_13_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpointsw
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
_temp/part
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
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: ü)
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:L*
dtype0*¥)
value)B)LB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-2/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-2/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-2/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-5/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-5/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/gamma/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/beta/.ATTRIBUTES/VARIABLE_VALUEB;layer_with_weights-7/moving_mean/.ATTRIBUTES/VARIABLE_VALUEB?layer_with_weights-7/moving_variance/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-9/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-9/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-2/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-5/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-5/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-6/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-6/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBQlayer_with_weights-7/gamma/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-7/beta/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-8/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-8/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-9/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-9/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:L*
dtype0*­
value£B LB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B  
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_34_kernel_read_readvariableop)savev2_conv2d_34_bias_read_readvariableop+savev2_conv2d_35_kernel_read_readvariableop)savev2_conv2d_35_bias_read_readvariableop7savev2_batch_normalization_18_gamma_read_readvariableop6savev2_batch_normalization_18_beta_read_readvariableop=savev2_batch_normalization_18_moving_mean_read_readvariableopAsavev2_batch_normalization_18_moving_variance_read_readvariableop+savev2_conv2d_36_kernel_read_readvariableop)savev2_conv2d_36_bias_read_readvariableop+savev2_conv2d_37_kernel_read_readvariableop)savev2_conv2d_37_bias_read_readvariableop7savev2_batch_normalization_19_gamma_read_readvariableop6savev2_batch_normalization_19_beta_read_readvariableop=savev2_batch_normalization_19_moving_mean_read_readvariableopAsavev2_batch_normalization_19_moving_variance_read_readvariableop+savev2_conv2d_38_kernel_read_readvariableop)savev2_conv2d_38_bias_read_readvariableop7savev2_batch_normalization_20_gamma_read_readvariableop6savev2_batch_normalization_20_beta_read_readvariableop=savev2_batch_normalization_20_moving_mean_read_readvariableopAsavev2_batch_normalization_20_moving_variance_read_readvariableop*savev2_dense_12_kernel_read_readvariableop(savev2_dense_12_bias_read_readvariableop*savev2_dense_13_kernel_read_readvariableop(savev2_dense_13_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop2savev2_adam_conv2d_34_kernel_m_read_readvariableop0savev2_adam_conv2d_34_bias_m_read_readvariableop2savev2_adam_conv2d_35_kernel_m_read_readvariableop0savev2_adam_conv2d_35_bias_m_read_readvariableop>savev2_adam_batch_normalization_18_gamma_m_read_readvariableop=savev2_adam_batch_normalization_18_beta_m_read_readvariableop2savev2_adam_conv2d_36_kernel_m_read_readvariableop0savev2_adam_conv2d_36_bias_m_read_readvariableop2savev2_adam_conv2d_37_kernel_m_read_readvariableop0savev2_adam_conv2d_37_bias_m_read_readvariableop>savev2_adam_batch_normalization_19_gamma_m_read_readvariableop=savev2_adam_batch_normalization_19_beta_m_read_readvariableop2savev2_adam_conv2d_38_kernel_m_read_readvariableop0savev2_adam_conv2d_38_bias_m_read_readvariableop>savev2_adam_batch_normalization_20_gamma_m_read_readvariableop=savev2_adam_batch_normalization_20_beta_m_read_readvariableop1savev2_adam_dense_12_kernel_m_read_readvariableop/savev2_adam_dense_12_bias_m_read_readvariableop1savev2_adam_dense_13_kernel_m_read_readvariableop/savev2_adam_dense_13_bias_m_read_readvariableop2savev2_adam_conv2d_34_kernel_v_read_readvariableop0savev2_adam_conv2d_34_bias_v_read_readvariableop2savev2_adam_conv2d_35_kernel_v_read_readvariableop0savev2_adam_conv2d_35_bias_v_read_readvariableop>savev2_adam_batch_normalization_18_gamma_v_read_readvariableop=savev2_adam_batch_normalization_18_beta_v_read_readvariableop2savev2_adam_conv2d_36_kernel_v_read_readvariableop0savev2_adam_conv2d_36_bias_v_read_readvariableop2savev2_adam_conv2d_37_kernel_v_read_readvariableop0savev2_adam_conv2d_37_bias_v_read_readvariableop>savev2_adam_batch_normalization_19_gamma_v_read_readvariableop=savev2_adam_batch_normalization_19_beta_v_read_readvariableop2savev2_adam_conv2d_38_kernel_v_read_readvariableop0savev2_adam_conv2d_38_bias_v_read_readvariableop>savev2_adam_batch_normalization_20_gamma_v_read_readvariableop=savev2_adam_batch_normalization_20_beta_v_read_readvariableop1savev2_adam_dense_12_kernel_v_read_readvariableop/savev2_adam_dense_12_bias_v_read_readvariableop1savev2_adam_dense_13_kernel_v_read_readvariableop/savev2_adam_dense_13_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *Z
dtypesP
N2L	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
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

identity_1Identity_1:output:0*·
_input_shapes¥
¢: :@:@:@@:@:@:@:@:@:@::::::::::::::
::	:: : : : : : : : : :@:@:@@:@:@:@:@::::::::::
::	::@:@:@@:@:@:@:@::::::::::
::	:: 2(
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
:@:!


_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::.*
(
_output_shapes
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	: 
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
:@:!+

_output_shapes	
::.,*
(
_output_shapes
::!-

_output_shapes	
::!.

_output_shapes	
::!/

_output_shapes	
::.0*
(
_output_shapes
::!1

_output_shapes	
::!2

_output_shapes	
::!3

_output_shapes	
::&4"
 
_output_shapes
:
:!5

_output_shapes	
::%6!

_output_shapes
:	: 7
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
:@:!?

_output_shapes	
::.@*
(
_output_shapes
::!A

_output_shapes	
::!B

_output_shapes	
::!C

_output_shapes	
::.D*
(
_output_shapes
::!E

_output_shapes	
::!F

_output_shapes	
::!G

_output_shapes	
::&H"
 
_output_shapes
:
:!I

_output_shapes	
::%J!

_output_shapes
:	: K

_output_shapes
::L

_output_shapes
: 

Æ
S__inference_batch_normalization_19_layer_call_and_return_conditional_losses_2431866

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Û
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<Æ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ð
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÔ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Â
S__inference_batch_normalization_18_layer_call_and_return_conditional_losses_2433121

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0Ö
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
exponential_avg_factor%
×#<Æ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ð
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(}
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@Ô
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs
¿
N
2__inference_max_pooling2d_18_layer_call_fn_2433054

inputs
identityÞ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_18_layer_call_and_return_conditional_losses_2431734
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

Æ
S__inference_batch_normalization_20_layer_call_and_return_conditional_losses_2433325

inputs&
readvariableop_resource:	(
readvariableop_1_resource:	7
(fusedbatchnormv3_readvariableop_resource:	9
*fusedbatchnormv3_readvariableop_1_resource:	
identity¢AssignNewValue¢AssignNewValue_1¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1c
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes	
:*
dtype0g
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes	
:*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes	
:*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes	
:*
dtype0Û
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*b
_output_shapesP
N:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:::::*
epsilon%o:*
exponential_avg_factor%
×#<Æ
AssignNewValueAssignVariableOp(fusedbatchnormv3_readvariableop_resourceFusedBatchNormV3:batch_mean:0 ^FusedBatchNormV3/ReadVariableOp*
_output_shapes
 *
dtype0*
validate_shape(Ð
AssignNewValue_1AssignVariableOp*fusedbatchnormv3_readvariableop_1_resource!FusedBatchNormV3:batch_variance:0"^FusedBatchNormV3/ReadVariableOp_1*
_output_shapes
 *
dtype0*
validate_shape(~
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÔ
NoOpNoOp^AssignNewValue^AssignNewValue_1 ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ: : : : 2 
AssignNewValueAssignNewValue2$
AssignNewValue_1AssignNewValue_12B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:j f
B
_output_shapes0
.:,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Î

S__inference_batch_normalization_18_layer_call_and_return_conditional_losses_2431759

inputs%
readvariableop_resource:@'
readvariableop_1_resource:@6
(fusedbatchnormv3_readvariableop_resource:@8
*fusedbatchnormv3_readvariableop_1_resource:@
identity¢FusedBatchNormV3/ReadVariableOp¢!FusedBatchNormV3/ReadVariableOp_1¢ReadVariableOp¢ReadVariableOp_1b
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:@*
dtype0f
ReadVariableOp_1ReadVariableOpreadvariableop_1_resource*
_output_shapes
:@*
dtype0
FusedBatchNormV3/ReadVariableOpReadVariableOp(fusedbatchnormv3_readvariableop_resource*
_output_shapes
:@*
dtype0
!FusedBatchNormV3/ReadVariableOp_1ReadVariableOp*fusedbatchnormv3_readvariableop_1_resource*
_output_shapes
:@*
dtype0È
FusedBatchNormV3FusedBatchNormV3inputsReadVariableOp:value:0ReadVariableOp_1:value:0'FusedBatchNormV3/ReadVariableOp:value:0)FusedBatchNormV3/ReadVariableOp_1:value:0*
T0*
U0*]
_output_shapesK
I:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@:@:@:@:@:*
epsilon%o:*
is_training( }
IdentityIdentityFusedBatchNormV3:y:0^NoOp*
T0*A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@°
NoOpNoOp ^FusedBatchNormV3/ReadVariableOp"^FusedBatchNormV3/ReadVariableOp_1^ReadVariableOp^ReadVariableOp_1*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*H
_input_shapes7
5:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@: : : : 2B
FusedBatchNormV3/ReadVariableOpFusedBatchNormV3/ReadVariableOp2F
!FusedBatchNormV3/ReadVariableOp_1!FusedBatchNormV3/ReadVariableOp_12 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_1:i e
A
_output_shapes/
-:+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

Ö
.__inference_sequential_8_layer_call_fn_2432809

inputs!
unknown:@
	unknown_0:@#
	unknown_1:@@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@
	unknown_6:@$
	unknown_7:@
	unknown_8:	%
	unknown_9:

unknown_10:	

unknown_11:	

unknown_12:	

unknown_13:	

unknown_14:	&

unknown_15:

unknown_16:	

unknown_17:	

unknown_18:	

unknown_19:	

unknown_20:	

unknown_21:


unknown_22:	

unknown_23:	

unknown_24:
identity¢StatefulPartitionedCall¢
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
:ÿÿÿÿÿÿÿÿÿ*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_sequential_8_layer_call_and_return_conditional_losses_2432378o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:ÿÿÿÿÿÿÿÿÿ8: : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8
 
_user_specified_nameinputs


F__inference_conv2d_36_layer_call_and_return_conditional_losses_2433141

inputs9
conv2d_readvariableop_resource:@.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

i
M__inference_max_pooling2d_18_layer_call_and_return_conditional_losses_2433059

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
I
È
I__inference_sequential_8_layer_call_and_return_conditional_losses_2432560
conv2d_34_input+
conv2d_34_2432493:@
conv2d_34_2432495:@+
conv2d_35_2432498:@@
conv2d_35_2432500:@,
batch_normalization_18_2432504:@,
batch_normalization_18_2432506:@,
batch_normalization_18_2432508:@,
batch_normalization_18_2432510:@,
conv2d_36_2432513:@ 
conv2d_36_2432515:	-
conv2d_37_2432518: 
conv2d_37_2432520:	-
batch_normalization_19_2432524:	-
batch_normalization_19_2432526:	-
batch_normalization_19_2432528:	-
batch_normalization_19_2432530:	-
conv2d_38_2432533: 
conv2d_38_2432535:	-
batch_normalization_20_2432539:	-
batch_normalization_20_2432541:	-
batch_normalization_20_2432543:	-
batch_normalization_20_2432545:	$
dense_12_2432549:

dense_12_2432551:	#
dense_13_2432554:	
dense_13_2432556:
identity¢.batch_normalization_18/StatefulPartitionedCall¢.batch_normalization_19/StatefulPartitionedCall¢.batch_normalization_20/StatefulPartitionedCall¢!conv2d_34/StatefulPartitionedCall¢!conv2d_35/StatefulPartitionedCall¢!conv2d_36/StatefulPartitionedCall¢!conv2d_37/StatefulPartitionedCall¢!conv2d_38/StatefulPartitionedCall¢ dense_12/StatefulPartitionedCall¢ dense_13/StatefulPartitionedCall
!conv2d_34/StatefulPartitionedCallStatefulPartitionedCallconv2d_34_inputconv2d_34_2432493conv2d_34_2432495*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ6@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_34_layer_call_and_return_conditional_losses_2431971¦
!conv2d_35/StatefulPartitionedCallStatefulPartitionedCall*conv2d_34/StatefulPartitionedCall:output:0conv2d_35_2432498conv2d_35_2432500*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_35_layer_call_and_return_conditional_losses_2431988ø
 max_pooling2d_18/PartitionedCallPartitionedCall*conv2d_35/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_18_layer_call_and_return_conditional_losses_2431734
.batch_normalization_18/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_18/PartitionedCall:output:0batch_normalization_18_2432504batch_normalization_18_2432506batch_normalization_18_2432508batch_normalization_18_2432510*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_18_layer_call_and_return_conditional_losses_2431759´
!conv2d_36/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_18/StatefulPartitionedCall:output:0conv2d_36_2432513conv2d_36_2432515*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_36_layer_call_and_return_conditional_losses_2432015§
!conv2d_37/StatefulPartitionedCallStatefulPartitionedCall*conv2d_36/StatefulPartitionedCall:output:0conv2d_37_2432518conv2d_37_2432520*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_37_layer_call_and_return_conditional_losses_2432032ù
 max_pooling2d_19/PartitionedCallPartitionedCall*conv2d_37/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_19_layer_call_and_return_conditional_losses_2431810
.batch_normalization_19/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_19/PartitionedCall:output:0batch_normalization_19_2432524batch_normalization_19_2432526batch_normalization_19_2432528batch_normalization_19_2432530*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_19_layer_call_and_return_conditional_losses_2431835´
!conv2d_38/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_19/StatefulPartitionedCall:output:0conv2d_38_2432533conv2d_38_2432535*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_38_layer_call_and_return_conditional_losses_2432059ù
 max_pooling2d_20/PartitionedCallPartitionedCall*conv2d_38/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_20_layer_call_and_return_conditional_losses_2431886
.batch_normalization_20/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_20/PartitionedCall:output:0batch_normalization_20_2432539batch_normalization_20_2432541batch_normalization_20_2432543batch_normalization_20_2432545*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_20_layer_call_and_return_conditional_losses_2431911ð
flatten_6/PartitionedCallPartitionedCall7batch_normalization_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_flatten_6_layer_call_and_return_conditional_losses_2432081
 dense_12/StatefulPartitionedCallStatefulPartitionedCall"flatten_6/PartitionedCall:output:0dense_12_2432549dense_12_2432551*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_12_layer_call_and_return_conditional_losses_2432094
 dense_13/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0dense_13_2432554dense_13_2432556*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_13_layer_call_and_return_conditional_losses_2432111x
IdentityIdentity)dense_13/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÓ
NoOpNoOp/^batch_normalization_18/StatefulPartitionedCall/^batch_normalization_19/StatefulPartitionedCall/^batch_normalization_20/StatefulPartitionedCall"^conv2d_34/StatefulPartitionedCall"^conv2d_35/StatefulPartitionedCall"^conv2d_36/StatefulPartitionedCall"^conv2d_37/StatefulPartitionedCall"^conv2d_38/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:ÿÿÿÿÿÿÿÿÿ8: : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_18/StatefulPartitionedCall.batch_normalization_18/StatefulPartitionedCall2`
.batch_normalization_19/StatefulPartitionedCall.batch_normalization_19/StatefulPartitionedCall2`
.batch_normalization_20/StatefulPartitionedCall.batch_normalization_20/StatefulPartitionedCall2F
!conv2d_34/StatefulPartitionedCall!conv2d_34/StatefulPartitionedCall2F
!conv2d_35/StatefulPartitionedCall!conv2d_35/StatefulPartitionedCall2F
!conv2d_36/StatefulPartitionedCall!conv2d_36/StatefulPartitionedCall2F
!conv2d_37/StatefulPartitionedCall!conv2d_37/StatefulPartitionedCall2F
!conv2d_38/StatefulPartitionedCall!conv2d_38/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall:` \
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8
)
_user_specified_nameconv2d_34_input
¿
N
2__inference_max_pooling2d_20_layer_call_fn_2433258

inputs
identityÞ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_20_layer_call_and_return_conditional_losses_2431886
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
¥

÷
E__inference_dense_13_layer_call_and_return_conditional_losses_2433376

inputs1
matmul_readvariableop_resource:	-
biasadd_readvariableop_resource:
identity¢BiasAdd/ReadVariableOp¢MatMul/ReadVariableOpu
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿV
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿw
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:ÿÿÿÿÿÿÿÿÿ: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


F__inference_conv2d_36_layer_call_and_return_conditional_losses_2432015

inputs9
conv2d_readvariableop_resource:@.
biasadd_readvariableop_resource:	
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp}
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*
paddingVALID*
strides
s
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype0~
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
Y
ReluReluBiasAdd:output:0*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
j
IdentityIdentityRelu:activations:0^NoOp*
T0*0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@
 
_user_specified_nameinputs

ÿ
F__inference_conv2d_35_layer_call_and_return_conditional_losses_2433049

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4@*
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
:ÿÿÿÿÿÿÿÿÿ4@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ6@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ6@
 
_user_specified_nameinputs
ìH
¿
I__inference_sequential_8_layer_call_and_return_conditional_losses_2432378

inputs+
conv2d_34_2432311:@
conv2d_34_2432313:@+
conv2d_35_2432316:@@
conv2d_35_2432318:@,
batch_normalization_18_2432322:@,
batch_normalization_18_2432324:@,
batch_normalization_18_2432326:@,
batch_normalization_18_2432328:@,
conv2d_36_2432331:@ 
conv2d_36_2432333:	-
conv2d_37_2432336: 
conv2d_37_2432338:	-
batch_normalization_19_2432342:	-
batch_normalization_19_2432344:	-
batch_normalization_19_2432346:	-
batch_normalization_19_2432348:	-
conv2d_38_2432351: 
conv2d_38_2432353:	-
batch_normalization_20_2432357:	-
batch_normalization_20_2432359:	-
batch_normalization_20_2432361:	-
batch_normalization_20_2432363:	$
dense_12_2432367:

dense_12_2432369:	#
dense_13_2432372:	
dense_13_2432374:
identity¢.batch_normalization_18/StatefulPartitionedCall¢.batch_normalization_19/StatefulPartitionedCall¢.batch_normalization_20/StatefulPartitionedCall¢!conv2d_34/StatefulPartitionedCall¢!conv2d_35/StatefulPartitionedCall¢!conv2d_36/StatefulPartitionedCall¢!conv2d_37/StatefulPartitionedCall¢!conv2d_38/StatefulPartitionedCall¢ dense_12/StatefulPartitionedCall¢ dense_13/StatefulPartitionedCall
!conv2d_34/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_34_2432311conv2d_34_2432313*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ6@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_34_layer_call_and_return_conditional_losses_2431971¦
!conv2d_35/StatefulPartitionedCallStatefulPartitionedCall*conv2d_34/StatefulPartitionedCall:output:0conv2d_35_2432316conv2d_35_2432318*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ4@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_35_layer_call_and_return_conditional_losses_2431988ø
 max_pooling2d_18/PartitionedCallPartitionedCall*conv2d_35/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_18_layer_call_and_return_conditional_losses_2431734
.batch_normalization_18/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_18/PartitionedCall:output:0batch_normalization_18_2432322batch_normalization_18_2432324batch_normalization_18_2432326batch_normalization_18_2432328*
Tin	
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:ÿÿÿÿÿÿÿÿÿ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_18_layer_call_and_return_conditional_losses_2431790´
!conv2d_36/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_18/StatefulPartitionedCall:output:0conv2d_36_2432331conv2d_36_2432333*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_36_layer_call_and_return_conditional_losses_2432015§
!conv2d_37/StatefulPartitionedCallStatefulPartitionedCall*conv2d_36/StatefulPartitionedCall:output:0conv2d_37_2432336conv2d_37_2432338*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_37_layer_call_and_return_conditional_losses_2432032ù
 max_pooling2d_19/PartitionedCallPartitionedCall*conv2d_37/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_19_layer_call_and_return_conditional_losses_2431810
.batch_normalization_19/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_19/PartitionedCall:output:0batch_normalization_19_2432342batch_normalization_19_2432344batch_normalization_19_2432346batch_normalization_19_2432348*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_19_layer_call_and_return_conditional_losses_2431866´
!conv2d_38/StatefulPartitionedCallStatefulPartitionedCall7batch_normalization_19/StatefulPartitionedCall:output:0conv2d_38_2432351conv2d_38_2432353*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ	*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_conv2d_38_layer_call_and_return_conditional_losses_2432059ù
 max_pooling2d_20/PartitionedCallPartitionedCall*conv2d_38/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_20_layer_call_and_return_conditional_losses_2431886
.batch_normalization_20/StatefulPartitionedCallStatefulPartitionedCall)max_pooling2d_20/PartitionedCall:output:0batch_normalization_20_2432357batch_normalization_20_2432359batch_normalization_20_2432361batch_normalization_20_2432363*
Tin	
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_batch_normalization_20_layer_call_and_return_conditional_losses_2431942ð
flatten_6/PartitionedCallPartitionedCall7batch_normalization_20/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_flatten_6_layer_call_and_return_conditional_losses_2432081
 dense_12/StatefulPartitionedCallStatefulPartitionedCall"flatten_6/PartitionedCall:output:0dense_12_2432367dense_12_2432369*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_12_layer_call_and_return_conditional_losses_2432094
 dense_13/StatefulPartitionedCallStatefulPartitionedCall)dense_12/StatefulPartitionedCall:output:0dense_13_2432372dense_13_2432374*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_dense_13_layer_call_and_return_conditional_losses_2432111x
IdentityIdentity)dense_13/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿÓ
NoOpNoOp/^batch_normalization_18/StatefulPartitionedCall/^batch_normalization_19/StatefulPartitionedCall/^batch_normalization_20/StatefulPartitionedCall"^conv2d_34/StatefulPartitionedCall"^conv2d_35/StatefulPartitionedCall"^conv2d_36/StatefulPartitionedCall"^conv2d_37/StatefulPartitionedCall"^conv2d_38/StatefulPartitionedCall!^dense_12/StatefulPartitionedCall!^dense_13/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*b
_input_shapesQ
O:ÿÿÿÿÿÿÿÿÿ8: : : : : : : : : : : : : : : : : : : : : : : : : : 2`
.batch_normalization_18/StatefulPartitionedCall.batch_normalization_18/StatefulPartitionedCall2`
.batch_normalization_19/StatefulPartitionedCall.batch_normalization_19/StatefulPartitionedCall2`
.batch_normalization_20/StatefulPartitionedCall.batch_normalization_20/StatefulPartitionedCall2F
!conv2d_34/StatefulPartitionedCall!conv2d_34/StatefulPartitionedCall2F
!conv2d_35/StatefulPartitionedCall!conv2d_35/StatefulPartitionedCall2F
!conv2d_36/StatefulPartitionedCall!conv2d_36/StatefulPartitionedCall2F
!conv2d_37/StatefulPartitionedCall!conv2d_37/StatefulPartitionedCall2F
!conv2d_38/StatefulPartitionedCall!conv2d_38/StatefulPartitionedCall2D
 dense_12/StatefulPartitionedCall dense_12/StatefulPartitionedCall2D
 dense_13/StatefulPartitionedCall dense_13/StatefulPartitionedCall:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8
 
_user_specified_nameinputs

ÿ
F__inference_conv2d_34_layer_call_and_return_conditional_losses_2431971

inputs8
conv2d_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity¢BiasAdd/ReadVariableOp¢Conv2D/ReadVariableOp|
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@*
dtype0
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ6@*
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
:ÿÿÿÿÿÿÿÿÿ6@X
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ6@i
IdentityIdentityRelu:activations:0^NoOp*
T0*/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ6@w
NoOpNoOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:ÿÿÿÿÿÿÿÿÿ8: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:ÿÿÿÿÿÿÿÿÿ8
 
_user_specified_nameinputs
¿
N
2__inference_max_pooling2d_19_layer_call_fn_2433166

inputs
identityÞ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *V
fQRO
M__inference_max_pooling2d_19_layer_call_and_return_conditional_losses_2431810
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

i
M__inference_max_pooling2d_19_layer_call_and_return_conditional_losses_2431810

inputs
identity¢
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ*
ksize
*
paddingVALID*
strides
{
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ:r n
J
_output_shapes8
6:4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"µ	L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ã
serving_default¯
S
conv2d_34_input@
!serving_default_conv2d_34_input:0ÿÿÿÿÿÿÿÿÿ8<
dense_130
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:Àð
¤
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
Ý
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
Ý
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
¥
*	variables
+trainable_variables
,regularization_losses
-	keras_api
.__call__
*/&call_and_return_all_conditional_losses"
_tf_keras_layer
ê
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
Ý
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
Ý
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
¥
M	variables
Ntrainable_variables
Oregularization_losses
P	keras_api
Q__call__
*R&call_and_return_all_conditional_losses"
_tf_keras_layer
ê
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
Ý
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
¥
g	variables
htrainable_variables
iregularization_losses
j	keras_api
k__call__
*l&call_and_return_all_conditional_losses"
_tf_keras_layer
ê
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
¥
x	variables
ytrainable_variables
zregularization_losses
{	keras_api
|__call__
*}&call_and_return_all_conditional_losses"
_tf_keras_layer
Á
~	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias"
_tf_keras_layer
Ã
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias"
_tf_keras_layer
ê
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
22
23
24
25"
trackable_list_wrapper
º
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
16
17
18
19"
trackable_list_wrapper
 "
trackable_list_wrapper
Ï
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
õ
trace_0
trace_1
trace_2
trace_32
.__inference_sequential_8_layer_call_fn_2432173
.__inference_sequential_8_layer_call_fn_2432752
.__inference_sequential_8_layer_call_fn_2432809
.__inference_sequential_8_layer_call_fn_2432490¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0ztrace_1ztrace_2ztrace_3
á
trace_0
trace_1
trace_2
trace_32î
I__inference_sequential_8_layer_call_and_return_conditional_losses_2432909
I__inference_sequential_8_layer_call_and_return_conditional_losses_2433009
I__inference_sequential_8_layer_call_and_return_conditional_losses_2432560
I__inference_sequential_8_layer_call_and_return_conditional_losses_2432630¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0ztrace_1ztrace_2ztrace_3
ÕBÒ
"__inference__wrapped_model_2431725conv2d_34_input"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ð
	iter
beta_1
beta_2

decay
learning_ratemm'm(m7m8mAmBmJmKmZm[mdm em¡tm¢um£	m¤	m¥	m¦	m§v¨v©'vª(v«7v¬8v­Av®Bv¯Jv°Kv±Zv²[v³dv´evµtv¶uv·	v¸	v¹	vº	v»"
	optimizer
-
 serving_default"
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
²
¡non_trainable_variables
¢layers
£metrics
 ¤layer_regularization_losses
¥layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ñ
¦trace_02Ò
+__inference_conv2d_34_layer_call_fn_2433018¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z¦trace_0

§trace_02í
F__inference_conv2d_34_layer_call_and_return_conditional_losses_2433029¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z§trace_0
*:(@2conv2d_34/kernel
:@2conv2d_34/bias
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
²
¨non_trainable_variables
©layers
ªmetrics
 «layer_regularization_losses
¬layer_metrics
!	variables
"trainable_variables
#regularization_losses
%__call__
*&&call_and_return_all_conditional_losses
&&"call_and_return_conditional_losses"
_generic_user_object
ñ
­trace_02Ò
+__inference_conv2d_35_layer_call_fn_2433038¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z­trace_0

®trace_02í
F__inference_conv2d_35_layer_call_and_return_conditional_losses_2433049¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z®trace_0
*:(@@2conv2d_35/kernel
:@2conv2d_35/bias
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
¯non_trainable_variables
°layers
±metrics
 ²layer_regularization_losses
³layer_metrics
*	variables
+trainable_variables
,regularization_losses
.__call__
*/&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses"
_generic_user_object
ø
´trace_02Ù
2__inference_max_pooling2d_18_layer_call_fn_2433054¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z´trace_0

µtrace_02ô
M__inference_max_pooling2d_18_layer_call_and_return_conditional_losses_2433059¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zµtrace_0
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
²
¶non_trainable_variables
·layers
¸metrics
 ¹layer_regularization_losses
ºlayer_metrics
0	variables
1trainable_variables
2regularization_losses
4__call__
*5&call_and_return_all_conditional_losses
&5"call_and_return_conditional_losses"
_generic_user_object
å
»trace_0
¼trace_12ª
8__inference_batch_normalization_18_layer_call_fn_2433072
8__inference_batch_normalization_18_layer_call_fn_2433085³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z»trace_0z¼trace_1

½trace_0
¾trace_12à
S__inference_batch_normalization_18_layer_call_and_return_conditional_losses_2433103
S__inference_batch_normalization_18_layer_call_and_return_conditional_losses_2433121³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 z½trace_0z¾trace_1
 "
trackable_list_wrapper
*:(@2batch_normalization_18/gamma
):'@2batch_normalization_18/beta
2:0@ (2"batch_normalization_18/moving_mean
6:4@ (2&batch_normalization_18/moving_variance
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
²
¿non_trainable_variables
Àlayers
Ámetrics
 Âlayer_regularization_losses
Ãlayer_metrics
;	variables
<trainable_variables
=regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses"
_generic_user_object
ñ
Ätrace_02Ò
+__inference_conv2d_36_layer_call_fn_2433130¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÄtrace_0

Åtrace_02í
F__inference_conv2d_36_layer_call_and_return_conditional_losses_2433141¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÅtrace_0
+:)@2conv2d_36/kernel
:2conv2d_36/bias
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
²
Ænon_trainable_variables
Çlayers
Èmetrics
 Élayer_regularization_losses
Êlayer_metrics
D	variables
Etrainable_variables
Fregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses"
_generic_user_object
ñ
Ëtrace_02Ò
+__inference_conv2d_37_layer_call_fn_2433150¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zËtrace_0

Ìtrace_02í
F__inference_conv2d_37_layer_call_and_return_conditional_losses_2433161¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÌtrace_0
,:*2conv2d_37/kernel
:2conv2d_37/bias
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
Ínon_trainable_variables
Îlayers
Ïmetrics
 Ðlayer_regularization_losses
Ñlayer_metrics
M	variables
Ntrainable_variables
Oregularization_losses
Q__call__
*R&call_and_return_all_conditional_losses
&R"call_and_return_conditional_losses"
_generic_user_object
ø
Òtrace_02Ù
2__inference_max_pooling2d_19_layer_call_fn_2433166¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÒtrace_0

Ótrace_02ô
M__inference_max_pooling2d_19_layer_call_and_return_conditional_losses_2433171¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÓtrace_0
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
²
Ônon_trainable_variables
Õlayers
Ömetrics
 ×layer_regularization_losses
Ølayer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
å
Ùtrace_0
Útrace_12ª
8__inference_batch_normalization_19_layer_call_fn_2433184
8__inference_batch_normalization_19_layer_call_fn_2433197³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÙtrace_0zÚtrace_1

Ûtrace_0
Ütrace_12à
S__inference_batch_normalization_19_layer_call_and_return_conditional_losses_2433215
S__inference_batch_normalization_19_layer_call_and_return_conditional_losses_2433233³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zÛtrace_0zÜtrace_1
 "
trackable_list_wrapper
+:)2batch_normalization_19/gamma
*:(2batch_normalization_19/beta
3:1 (2"batch_normalization_19/moving_mean
7:5 (2&batch_normalization_19/moving_variance
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
²
Ýnon_trainable_variables
Þlayers
ßmetrics
 àlayer_regularization_losses
álayer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
ñ
âtrace_02Ò
+__inference_conv2d_38_layer_call_fn_2433242¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zâtrace_0

ãtrace_02í
F__inference_conv2d_38_layer_call_and_return_conditional_losses_2433253¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zãtrace_0
,:*2conv2d_38/kernel
:2conv2d_38/bias
´2±®
£²
FullArgSpec'
args
jself
jinputs
jkernel
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
änon_trainable_variables
ålayers
æmetrics
 çlayer_regularization_losses
èlayer_metrics
g	variables
htrainable_variables
iregularization_losses
k__call__
*l&call_and_return_all_conditional_losses
&l"call_and_return_conditional_losses"
_generic_user_object
ø
étrace_02Ù
2__inference_max_pooling2d_20_layer_call_fn_2433258¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zétrace_0

êtrace_02ô
M__inference_max_pooling2d_20_layer_call_and_return_conditional_losses_2433263¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zêtrace_0
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
²
ënon_trainable_variables
ìlayers
ímetrics
 îlayer_regularization_losses
ïlayer_metrics
m	variables
ntrainable_variables
oregularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
å
ðtrace_0
ñtrace_12ª
8__inference_batch_normalization_20_layer_call_fn_2433276
8__inference_batch_normalization_20_layer_call_fn_2433289³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zðtrace_0zñtrace_1

òtrace_0
ótrace_12à
S__inference_batch_normalization_20_layer_call_and_return_conditional_losses_2433307
S__inference_batch_normalization_20_layer_call_and_return_conditional_losses_2433325³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zòtrace_0zótrace_1
 "
trackable_list_wrapper
+:)2batch_normalization_20/gamma
*:(2batch_normalization_20/beta
3:1 (2"batch_normalization_20/moving_mean
7:5 (2&batch_normalization_20/moving_variance
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
²
ônon_trainable_variables
õlayers
ömetrics
 ÷layer_regularization_losses
ølayer_metrics
x	variables
ytrainable_variables
zregularization_losses
|__call__
*}&call_and_return_all_conditional_losses
&}"call_and_return_conditional_losses"
_generic_user_object
ñ
ùtrace_02Ò
+__inference_flatten_6_layer_call_fn_2433330¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zùtrace_0

útrace_02í
F__inference_flatten_6_layer_call_and_return_conditional_losses_2433336¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 zútrace_0
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¶
ûnon_trainable_variables
ülayers
ýmetrics
 þlayer_regularization_losses
ÿlayer_metrics
~	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ð
trace_02Ñ
*__inference_dense_12_layer_call_fn_2433345¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0

trace_02ì
E__inference_dense_12_layer_call_and_return_conditional_losses_2433356¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0
#:!
2dense_12/kernel
:2dense_12/bias
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
ð
trace_02Ñ
*__inference_dense_13_layer_call_fn_2433365¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0

trace_02ì
E__inference_dense_13_layer_call_and_return_conditional_losses_2433376¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 ztrace_0
": 	2dense_13/kernel
:2dense_13/bias
J
90
:1
\2
]3
v4
w5"
trackable_list_wrapper

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
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
B
.__inference_sequential_8_layer_call_fn_2432173conv2d_34_input"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÿBü
.__inference_sequential_8_layer_call_fn_2432752inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ÿBü
.__inference_sequential_8_layer_call_fn_2432809inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
.__inference_sequential_8_layer_call_fn_2432490conv2d_34_input"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
I__inference_sequential_8_layer_call_and_return_conditional_losses_2432909inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
I__inference_sequential_8_layer_call_and_return_conditional_losses_2433009inputs"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
£B 
I__inference_sequential_8_layer_call_and_return_conditional_losses_2432560conv2d_34_input"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
£B 
I__inference_sequential_8_layer_call_and_return_conditional_losses_2432630conv2d_34_input"¿
¶²²
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
ÔBÑ
%__inference_signature_wrapper_2432695conv2d_34_input"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
ßBÜ
+__inference_conv2d_34_layer_call_fn_2433018inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
úB÷
F__inference_conv2d_34_layer_call_and_return_conditional_losses_2433029inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
ßBÜ
+__inference_conv2d_35_layer_call_fn_2433038inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
úB÷
F__inference_conv2d_35_layer_call_and_return_conditional_losses_2433049inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
æBã
2__inference_max_pooling2d_18_layer_call_fn_2433054inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Bþ
M__inference_max_pooling2d_18_layer_call_and_return_conditional_losses_2433059inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
ýBú
8__inference_batch_normalization_18_layer_call_fn_2433072inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ýBú
8__inference_batch_normalization_18_layer_call_fn_2433085inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
S__inference_batch_normalization_18_layer_call_and_return_conditional_losses_2433103inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
S__inference_batch_normalization_18_layer_call_and_return_conditional_losses_2433121inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
ßBÜ
+__inference_conv2d_36_layer_call_fn_2433130inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
úB÷
F__inference_conv2d_36_layer_call_and_return_conditional_losses_2433141inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
ßBÜ
+__inference_conv2d_37_layer_call_fn_2433150inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
úB÷
F__inference_conv2d_37_layer_call_and_return_conditional_losses_2433161inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
æBã
2__inference_max_pooling2d_19_layer_call_fn_2433166inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Bþ
M__inference_max_pooling2d_19_layer_call_and_return_conditional_losses_2433171inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
ýBú
8__inference_batch_normalization_19_layer_call_fn_2433184inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ýBú
8__inference_batch_normalization_19_layer_call_fn_2433197inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
S__inference_batch_normalization_19_layer_call_and_return_conditional_losses_2433215inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
S__inference_batch_normalization_19_layer_call_and_return_conditional_losses_2433233inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
ßBÜ
+__inference_conv2d_38_layer_call_fn_2433242inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
úB÷
F__inference_conv2d_38_layer_call_and_return_conditional_losses_2433253inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
æBã
2__inference_max_pooling2d_20_layer_call_fn_2433258inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
Bþ
M__inference_max_pooling2d_20_layer_call_and_return_conditional_losses_2433263inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
ýBú
8__inference_batch_normalization_20_layer_call_fn_2433276inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ýBú
8__inference_batch_normalization_20_layer_call_fn_2433289inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
S__inference_batch_normalization_20_layer_call_and_return_conditional_losses_2433307inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
B
S__inference_batch_normalization_20_layer_call_and_return_conditional_losses_2433325inputs"³
ª²¦
FullArgSpec)
args!
jself
jinputs

jtraining
varargs
 
varkw
 
defaults
p 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
ßBÜ
+__inference_flatten_6_layer_call_fn_2433330inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
úB÷
F__inference_flatten_6_layer_call_and_return_conditional_losses_2433336inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
ÞBÛ
*__inference_dense_12_layer_call_fn_2433345inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ùBö
E__inference_dense_12_layer_call_and_return_conditional_losses_2433356inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
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
ÞBÛ
*__inference_dense_13_layer_call_fn_2433365inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
ùBö
E__inference_dense_13_layer_call_and_return_conditional_losses_2433376inputs"¢
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *
 
R
	variables
	keras_api

total

count"
_tf_keras_metric
c
	variables
	keras_api

total

count

_fn_kwargs"
_tf_keras_metric
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
0
0
1"
trackable_list_wrapper
.
	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
/:-@2Adam/conv2d_34/kernel/m
!:@2Adam/conv2d_34/bias/m
/:-@@2Adam/conv2d_35/kernel/m
!:@2Adam/conv2d_35/bias/m
/:-@2#Adam/batch_normalization_18/gamma/m
.:,@2"Adam/batch_normalization_18/beta/m
0:.@2Adam/conv2d_36/kernel/m
": 2Adam/conv2d_36/bias/m
1:/2Adam/conv2d_37/kernel/m
": 2Adam/conv2d_37/bias/m
0:.2#Adam/batch_normalization_19/gamma/m
/:-2"Adam/batch_normalization_19/beta/m
1:/2Adam/conv2d_38/kernel/m
": 2Adam/conv2d_38/bias/m
0:.2#Adam/batch_normalization_20/gamma/m
/:-2"Adam/batch_normalization_20/beta/m
(:&
2Adam/dense_12/kernel/m
!:2Adam/dense_12/bias/m
':%	2Adam/dense_13/kernel/m
 :2Adam/dense_13/bias/m
/:-@2Adam/conv2d_34/kernel/v
!:@2Adam/conv2d_34/bias/v
/:-@@2Adam/conv2d_35/kernel/v
!:@2Adam/conv2d_35/bias/v
/:-@2#Adam/batch_normalization_18/gamma/v
.:,@2"Adam/batch_normalization_18/beta/v
0:.@2Adam/conv2d_36/kernel/v
": 2Adam/conv2d_36/bias/v
1:/2Adam/conv2d_37/kernel/v
": 2Adam/conv2d_37/bias/v
0:.2#Adam/batch_normalization_19/gamma/v
/:-2"Adam/batch_normalization_19/beta/v
1:/2Adam/conv2d_38/kernel/v
": 2Adam/conv2d_38/bias/v
0:.2#Adam/batch_normalization_20/gamma/v
/:-2"Adam/batch_normalization_20/beta/v
(:&
2Adam/dense_12/kernel/v
!:2Adam/dense_12/bias/v
':%	2Adam/dense_13/kernel/v
 :2Adam/dense_13/bias/v¾
"__inference__wrapped_model_2431725'(789:ABJKZ[\]detuvw@¢=
6¢3
1.
conv2d_34_inputÿÿÿÿÿÿÿÿÿ8
ª "3ª0
.
dense_13"
dense_13ÿÿÿÿÿÿÿÿÿî
S__inference_batch_normalization_18_layer_call_and_return_conditional_losses_2433103789:M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 î
S__inference_batch_normalization_18_layer_call_and_return_conditional_losses_2433121789:M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "?¢<
52
0+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
 Æ
8__inference_batch_normalization_18_layer_call_fn_2433072789:M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p 
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@Æ
8__inference_batch_normalization_18_layer_call_fn_2433085789:M¢J
C¢@
:7
inputs+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@
p
ª "2/+ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ@ð
S__inference_batch_normalization_19_layer_call_and_return_conditional_losses_2433215Z[\]N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ð
S__inference_batch_normalization_19_layer_call_and_return_conditional_losses_2433233Z[\]N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 È
8__inference_batch_normalization_19_layer_call_fn_2433184Z[\]N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
8__inference_batch_normalization_19_layer_call_fn_2433197Z[\]N¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿð
S__inference_batch_normalization_20_layer_call_and_return_conditional_losses_2433307tuvwN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 ð
S__inference_batch_normalization_20_layer_call_and_return_conditional_losses_2433325tuvwN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "@¢=
63
0,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 È
8__inference_batch_normalization_20_layer_call_fn_2433276tuvwN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p 
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÈ
8__inference_batch_normalization_20_layer_call_fn_2433289tuvwN¢K
D¢A
;8
inputs,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
p
ª "30,ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ¶
F__inference_conv2d_34_layer_call_and_return_conditional_losses_2433029l7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ8
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ6@
 
+__inference_conv2d_34_layer_call_fn_2433018_7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ8
ª " ÿÿÿÿÿÿÿÿÿ6@¶
F__inference_conv2d_35_layer_call_and_return_conditional_losses_2433049l'(7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ6@
ª "-¢*
# 
0ÿÿÿÿÿÿÿÿÿ4@
 
+__inference_conv2d_35_layer_call_fn_2433038_'(7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ6@
ª " ÿÿÿÿÿÿÿÿÿ4@·
F__inference_conv2d_36_layer_call_and_return_conditional_losses_2433141mAB7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ

 
+__inference_conv2d_36_layer_call_fn_2433130`AB7¢4
-¢*
(%
inputsÿÿÿÿÿÿÿÿÿ@
ª "!ÿÿÿÿÿÿÿÿÿ
¸
F__inference_conv2d_37_layer_call_and_return_conditional_losses_2433161nJK8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ

ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ
 
+__inference_conv2d_37_layer_call_fn_2433150aJK8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ

ª "!ÿÿÿÿÿÿÿÿÿ¸
F__inference_conv2d_38_layer_call_and_return_conditional_losses_2433253nde8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª ".¢+
$!
0ÿÿÿÿÿÿÿÿÿ	
 
+__inference_conv2d_38_layer_call_fn_2433242ade8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "!ÿÿÿÿÿÿÿÿÿ	©
E__inference_dense_12_layer_call_and_return_conditional_losses_2433356`0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
*__inference_dense_12_layer_call_fn_2433345S0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¨
E__inference_dense_13_layer_call_and_return_conditional_losses_2433376_0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
*__inference_dense_13_layer_call_fn_2433365R0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¬
F__inference_flatten_6_layer_call_and_return_conditional_losses_2433336b8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 
+__inference_flatten_6_layer_call_fn_2433330U8¢5
.¢+
)&
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿð
M__inference_max_pooling2d_18_layer_call_and_return_conditional_losses_2433059R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 È
2__inference_max_pooling2d_18_layer_call_fn_2433054R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿð
M__inference_max_pooling2d_19_layer_call_and_return_conditional_losses_2433171R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 È
2__inference_max_pooling2d_19_layer_call_fn_2433166R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿð
M__inference_max_pooling2d_20_layer_call_and_return_conditional_losses_2433263R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª "H¢E
>;
04ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
 È
2__inference_max_pooling2d_20_layer_call_fn_2433258R¢O
H¢E
C@
inputs4ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿ
ª ";84ÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿÿß
I__inference_sequential_8_layer_call_and_return_conditional_losses_2432560'(789:ABJKZ[\]detuvwH¢E
>¢;
1.
conv2d_34_inputÿÿÿÿÿÿÿÿÿ8
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ß
I__inference_sequential_8_layer_call_and_return_conditional_losses_2432630'(789:ABJKZ[\]detuvwH¢E
>¢;
1.
conv2d_34_inputÿÿÿÿÿÿÿÿÿ8
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ö
I__inference_sequential_8_layer_call_and_return_conditional_losses_2432909'(789:ABJKZ[\]detuvw?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ8
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ö
I__inference_sequential_8_layer_call_and_return_conditional_losses_2433009'(789:ABJKZ[\]detuvw?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ8
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ·
.__inference_sequential_8_layer_call_fn_2432173'(789:ABJKZ[\]detuvwH¢E
>¢;
1.
conv2d_34_inputÿÿÿÿÿÿÿÿÿ8
p 

 
ª "ÿÿÿÿÿÿÿÿÿ·
.__inference_sequential_8_layer_call_fn_2432490'(789:ABJKZ[\]detuvwH¢E
>¢;
1.
conv2d_34_inputÿÿÿÿÿÿÿÿÿ8
p

 
ª "ÿÿÿÿÿÿÿÿÿ­
.__inference_sequential_8_layer_call_fn_2432752{'(789:ABJKZ[\]detuvw?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ8
p 

 
ª "ÿÿÿÿÿÿÿÿÿ­
.__inference_sequential_8_layer_call_fn_2432809{'(789:ABJKZ[\]detuvw?¢<
5¢2
(%
inputsÿÿÿÿÿÿÿÿÿ8
p

 
ª "ÿÿÿÿÿÿÿÿÿÔ
%__inference_signature_wrapper_2432695ª'(789:ABJKZ[\]detuvwS¢P
¢ 
IªF
D
conv2d_34_input1.
conv2d_34_inputÿÿÿÿÿÿÿÿÿ8"3ª0
.
dense_13"
dense_13ÿÿÿÿÿÿÿÿÿ