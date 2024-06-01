/*
 * Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
 * with the License. A copy of the License is located at
 *
 * http://aws.amazon.com/apache2.0/
 *
 * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
 * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
 * and limitations under the License.
 */
@file:OptIn(ExperimentalUnsignedTypes::class)

package ai.djl.ndarray

import ai.djl.Device
import ai.djl.ndarray.index.NDIndex
import ai.djl.ndarray.internal.NDArrayEx
import ai.djl.ndarray.internal.NDFormat
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.ndarray.types.SparseFormat
import ai.djl.util.Float16Utils
import java.nio.*
import java.nio.charset.Charset
import java.nio.charset.StandardCharsets
import java.util.*
import java.util.function.Function
import java.util.stream.IntStream
import kotlin.math.abs

/**
 * An interface representing an n-dimensional array.
 *
 *
 * NDArray is the core data structure for all mathematical computations. An NDArray represents a
 * multidimensional, fixed-size homogeneous array. It has very similar behaviour to the Numpy python
 * package with the addition of efficient computing. To understand how to manage NDArray lifecycle,
 * please refer to [NDArray
 * Memory Management Guide](https://github.com/deepjavalibrary/djl/blob/master/docs/development/memory_management.md)
 */
interface NDArray : NDResource, BytesSupplier {
    /**
     * Returns the name of this `NDArray`.
     *
     * @return the name of this `NDArray`
     */
    /**
     * Sets name of this `NDArray`.
     *
     * @param name the name of this `NDArray`
     */
    //    @JvmField
    var name: String

    /**
     * Returns unique identifier of this `NDArray`.
     *
     * @return unique identifier of this `NDArray`
     */
    val uid: String

    /**
     * Returns the [DataType] of this `NDArray`.
     *
     *
     * [DataType] is a definition of the precision level of the `NDArray`. All values
     * inside the same `NDArray` would have the same [DataType].
     *
     * @return the [DataType] of this `NDArray`
     */
    //        @JvmField
    val dataType: DataType

    /**
     * Returns the [Device] of this `NDArray`.
     *
     *
     * [Device] class contains the information where this `NDArray` stored in memory,
     * like CPU/GPU.
     *
     * @return the [Device] of this `NDArray`
     */
    //    @JvmField
    val device: Device

    /**
     * Returns the [Shape] of this `NDArray`.
     *
     *
     * [Shape] defines how this `NDArray` is represented multi-dimensionally.
     *
     * @return the [Shape] of this `NDArray`
     */
    //    @JvmField
    val shape: Shape

    /**
     * Returns the [SparseFormat] of this `NDArray`.
     *
     * @return the [SparseFormat] of this `NDArray`
     */
    //    @JvmField
    val sparseFormat: SparseFormat

    val isSparse: Boolean
        /**
         * Returns `true` if this `NDArray` is a [SparseNDArray].
         *
         * @return `true` if this `NDArray` is a [SparseNDArray]
         */
        get() = sparseFormat != SparseFormat.DENSE

    val isScalar: Boolean
        /**
         * Returns `true` if this `NDArray` is a scalar `NDArray` with empty [ ].
         *
         * @return `true` if this `NDArray` is a scalar `NDArray` with empty [     ]
         */
        get() = shape.isScalar

    /**
     * Encodes `NDArray` to byte array.
     *
     * @return byte array
     */
    fun encode(): ByteArray = NDSerializer.encode(this)

    /**
     * Moves this `NDArray` to a different [Device].
     *
     * @param device the [Device] to be set
     * @param copy set `true` if you want to return a copy of the Existing `NDArray`
     * @return the result `NDArray` with the new [Device]
     */
    fun toDevice(device: Device, copy: Boolean): NDArray?

    /**
     * Converts this `NDArray` to a different [DataType].
     *
     * @param dataType the [DataType] to be set
     * @param copy set `true` if you want to return a copy of the Existing `NDArray`
     * @return the result `NDArray` with the new [DataType]
     */
    fun toType(dataType: DataType, copy: Boolean): NDArray

    /**
     * Attaches a gradient `NDArray` to this `NDArray` and marks it so [ ][ai.djl.training.GradientCollector.backward] can compute the gradient with respect to
     * it.
     *
     * @param requiresGrad if `NDArray` requires gradient or not
     */
    fun setRequiresGradient(requiresGrad: Boolean)

    /**
     * Returns the gradient `NDArray` attached to this `NDArray`.
     *
     * @return the gradient `NDArray`
     * @throws NullPointerException when gradient is not initialized
     */
    //    @JvmField
    val gradient: NDArray?

    /**
     * Returns true if the gradient calculation is required for this `NDArray`.
     *
     * @return true if the gradient calculation is required for this `NDArray` else false
     */
    fun hasGradient(): Boolean

    /**
     * Returns an NDArray equal to this that stop gradient propagation through it.
     *
     * @return an NDArray equal to this that stops gradient propagation through it
     */
    fun stopGradient(): NDArray

    /**
     * Returns an NDArray equal to this that magnifies the gradient propagated to this by a
     * constant.
     *
     * @param scale how much to magnify the gradient propagated to this
     * @return an NDArray equal to this that magnifies the gradient propagated to this by a constant
     */
    fun scaleGradient(scale: Double): NDArray = times(scale).plus(stopGradient().times(1 - scale))

    /**
     * Returns the size of this `NDArray` along a given axis.
     *
     * @param axis the axis to return the size for
     * @return the size of this `NDArray` along a given axis
     */
    infix fun size(axis: Int): Long = shape.size(axis)

    /**
     * Returns the total number of elements in this `NDArray`.
     *
     * @return the number of elements in this `NDArray`
     */
    val size: Long
        // https://youtrack.jetbrains.com/issue/KT-31420
        @Suppress("INAPPLICABLE_JVM_NAME") @JvmName("size")
        get() = shape.size()

    /** {@inheritDoc}  */
    override fun toByteBuffer(): ByteBuffer = toByteBuffer(false)

    /**
     * Returns the `ByteBuffer` presentation of the object.
     *
     *
     * If returned ByteBuffer is a DirectByteBuffer, it shared the same native memory as the
     * NDArray. The native memory will be deleted when NDArray is closed.
     *
     *
     * Not all the engine support return DirectByteBuffer.
     *
     * @param tryDirect use DirectBuffer if possible
     * @return the `ByteBuffer` presentation of the object
     */
    fun toByteBuffer(tryDirect: Boolean): ByteBuffer

    /**
     * Converts this `NDArray` to a double array.
     *
     * @return a double array
     * @throws IllegalStateException when [DataType] of this `NDArray` mismatches
     */
    fun toDoubleArray(): DoubleArray {
        check(dataType == DataType.FLOAT64) { "DataType mismatch, Required double Actual $dataType" }
        val db = toByteBuffer(true).asDoubleBuffer()
        return DoubleArray(db.remaining()).also(db::get)
    }

    /**
     * Converts this `NDArray` to a float array.
     *
     * @return a float array
     * @throws IllegalStateException when [DataType] of this `NDArray` mismatches
     */
    fun toFloatArray(): FloatArray {
        if (dataType == DataType.FLOAT16)
            return Float16Utils.fromByteBuffer(toByteBuffer())
        else
            check(dataType == DataType.FLOAT32) { "DataType mismatch, Required float, Actual $dataType" }
        val fb = toByteBuffer(true).asFloatBuffer()
        return FloatArray(fb.remaining()).also(fb::get)
    }

    /**
     * Converts this `NDArray` to a short array.
     *
     * @return an int array
     * @throws IllegalStateException when [DataType] of this `NDArray` mismatches
     */
    fun toShortArray(): ShortArray {
        check(dataType == DataType.INT16) { "DataType mismatch, Required int Actual $dataType" }
        val ib = toByteBuffer(true).asShortBuffer()
        return ShortArray(ib.remaining()).also(ib::get)
    }

    /**
     * Converts this `NDArray` to a short array.
     *
     * @return an int array
     * @throws IllegalStateException when [DataType] of this `NDArray` mismatches
     */
    fun toUShortArray(): UShortArray {
        check(dataType == DataType.UINT16) { "DataType mismatch, Required int Actual $dataType" }
        val ib = toByteBuffer(true).asShortBuffer()
        return UShortArray(ib.remaining()) { ib.get().toUShort() }
    }

    /**
     * Converts this `NDArray` to an int array.
     *
     * @return an int array
     * @throws IllegalStateException when [DataType] of this `NDArray` mismatches
     */
    fun toIntArray(): IntArray {
        val dType = dataType
        check(dType == DataType.INT32) { "DataType mismatch, Required int Actual $dataType" }
        val ib = toByteBuffer(true).asIntBuffer()
        return IntArray(ib.remaining()).also(ib::get)
    }

    /**
     * Converts this `NDArray` to an unsigned int array.
     *
     * @return an unsigned int array
     * @throws IllegalStateException when [DataType] of this `NDArray` mismatches
     */
    fun toUIntArray(): UIntArray {
        check(dataType == DataType.UINT32) { "DataType mismatch, Required int Actual $dataType" }
        val ib = toByteBuffer(true).asIntBuffer()
        return UIntArray(ib.remaining()) { ib.get().toUInt() }
    }

    /**
     * Converts this `NDArray` to a long array.
     *
     * @return a long array
     * @throws IllegalStateException when [DataType] of this `NDArray` mismatches
     */
    fun toLongArray(): LongArray {
        check(dataType == DataType.INT64) { "DataType mismatch, Required long Actual $dataType" }
        val lb = toByteBuffer(true).asLongBuffer()
        val ret = LongArray(lb.remaining())
        lb[ret]
        return ret
    }

    /**
     * Converts this `NDArray` to an unsigned long array.
     *
     * @return an unsigned long array
     * @throws IllegalStateException when [DataType] of this `NDArray` mismatches
     */
    fun toULongArray(): ULongArray {
        check(dataType == DataType.UINT64) { "DataType mismatch, Required long Actual $dataType" }
        val lb = toByteBuffer(true).asLongBuffer()
        return ULongArray(lb.remaining()) { lb.get().toULong() }
    }

    /**
     * Converts this `NDArray` to a byte array.
     *
     * @return a byte array
     * @throws IllegalStateException when [DataType] of this `NDArray` mismatches
     */
    fun toByteArray(): ByteArray {
        val bb = toByteBuffer(true)
        if (bb.hasArray() && bb.remaining() == bb.array().size) {
            return bb.array()
        }
        return ByteArray(bb.remaining()).also(bb::get)
    }

    /**
     * Converts this `NDArray` to an uint8 array.
     *
     * @return a uint8 array
     * @throws IllegalStateException when [DataType] of this `NDArray` mismatches
     */
    fun toUint8Array(): UByteArray {
        val bb = toByteBuffer(true)
        // TODO, for some reasons, `UByteArray` will trigger in `:compileJava`, so we add a trail `.toByteArray()`
        // error: cannot find symbol
        //        byte[] raw = array.toType(DataType.UINT8, false).toUint8Array();
        return UByteArray(bb.remaining()) { bb.get().toUByte() }/*.toByteArray()*/
    }

    /**
     * Converts this `NDArray` to a boolean array.
     *
     * @return a boolean array
     * @throws IllegalStateException when [DataType] of this `NDArray` mismatches
     */
    fun toBooleanArray(): BooleanArray {
        check(dataType == DataType.BOOLEAN) { "DataType mismatch, Required boolean Actual $dataType" }
        val bb = toByteBuffer(true)
        return BooleanArray(bb.remaining()) { bb.get().toInt() != 0 }
    }

    /**
     * Converts this `NDArray` to a String array.
     *
     *
     * This method is only applicable to the String typed NDArray and not for printing purpose
     *
     * @return Array of Strings
     */
    fun toStringArray(): Array<String> = toStringArray(StandardCharsets.UTF_8)

    /**
     * Converts this `NDArray` to a String array with the specified charset.
     *
     *
     * This method is only applicable to the String typed NDArray and not for printing purpose
     *
     * @param charset to charset for the string
     * @return Array of Strings
     */
    fun toStringArray(charset: Charset?): Array<String>

    /**
     * Converts this `NDArray` to a Number array based on its [DataType].
     *
     * @return a Number array
     */
    @Suppress("UNCHECKED_CAST")
    fun toArray(): Array<Number> = when (dataType) {
        DataType.FLOAT16, DataType.FLOAT32 -> toFloatArray().toTypedArray()
        DataType.FLOAT64 -> toDoubleArray().toTypedArray()
        DataType.INT16 -> toShortArray().toTypedArray()
        DataType.UINT16 -> toUShortArray().toTypedArray()
        DataType.INT32 -> toIntArray().toTypedArray()
        DataType.UINT32 -> toUIntArray().toTypedArray()
        DataType.INT64 -> toLongArray().toTypedArray()
        DataType.UINT64 -> toULongArray().toTypedArray()
        DataType.BOOLEAN, DataType.INT8 -> toByteArray().toTypedArray()
        DataType.UINT8 -> toUint8Array().toTypedArray()
        else -> throw IllegalStateException("Unsupported DataType: $dataType")
    } as Array<Number>

    /**
     * Sets this `NDArray` value from [Buffer].
     *
     * @param buffer the input buffered data
     */
    fun setFrom(buffer: Buffer)

    /**
     * Sets this `NDArray` value from an array of floats.
     *
     * @param data the array of floats to set
     */
    fun setFrom(data: FloatArray) = setFrom(FloatBuffer.wrap(data))

    /**
     * Sets this `NDArray` value from an array of ints.
     *
     * @param data the array of integers to set
     */
    fun setFrom(data: IntArray) = setFrom(IntBuffer.wrap(data))

    /**
     * Sets this `NDArray` value from an array of doubles.
     *
     * @param data the array of doubles to set
     */
    fun setFrom(data: DoubleArray) = setFrom(DoubleBuffer.wrap(data))

    /**
     * Sets this `NDArray` value from an array of longs.
     *
     * @param data the array of longs to set
     */
    fun setFrom(data: LongArray) = setFrom(LongBuffer.wrap(data))

    /**
     * Sets this `NDArray` value from an array of bytes.
     *
     * @param data the array of bytes to set
     */
    fun setFrom(data: ByteArray) = setFrom(ByteBuffer.wrap(data))

    /**
     * Sets the specified index in this `NDArray` with the given values.
     *
     * @param index the locations to update
     * @param value the value to replace with. Can broadcast if given smaller dimensions than the
     * index
     */
    operator fun set(index: NDIndex, value: NDArray) {
        nDArrayInternal.getIndexer(manager)[this, index] = value
    }

    /**
     * Sets the specified index in this `NDArray` with the given value.
     *
     * @param index the locations to update
     * @param value the value to replace with
     */
    operator fun set(index: NDIndex, value: Number) {
        nDArrayInternal.getIndexer(manager)[this, index] = value
    }

    /**
     * Sets the specific index by a function.
     *
     * @param index the locations to update
     * @param function the function to change the value
     */
    operator fun set(index: NDIndex, function: (NDArray) -> NDArray) {
        val array = this[index]
        this[index] = function(array)
    }

    /**
     * Sets the specific index by a function.
     *
     * @param index the locations to update
     * @param function the function to change the value
     */
    operator fun set(index: NDIndex, function: Function<NDArray, NDArray>) {
        val array = this[index]
        this[index] = function.apply(array)
    }

    /**
     * Sets the `NDArray` by boolean mask or integer index.
     *
     * @param index the boolean or integer `NDArray` that indicates what to get
     * @param value the value to replace with
     */
    operator fun set(index: NDArray, value: Number) {
        this[NDIndex("{}", index)] = value
    }

    /**
     * Sets the specified scalar in this `NDArray` with the given value.
     *
     * @param index the single index to update
     * @param value the value to replace with
     * @throws IllegalArgumentException thrown if the index does not correspond to a single element
     */
    fun setScalar(index: NDIndex, value: Number) = nDArrayInternal.getIndexer(manager).setScalar(this, index, value)

    /**
     * Returns a partial `NDArray`.
     *
     * @param index the section of this `NDArray` to return
     * @return the partial `NDArray`
     */
    operator fun get(index: NDIndex): NDArray = this[manager, index]

    /**
     * Returns a partial `NDArray`.
     *
     * @param manager the manager used to create the arrays
     * @param index the section of this `NDArray` to return
     * @return the partial `NDArray`
     */
    operator fun get(manager: NDManager, index: NDIndex): NDArray = nDArrayInternal.getIndexer(manager)[this, index]

    /**
     * Returns a partial `NDArray`.
     *
     * @param index the boolean or integer `NDArray` that indicates what to get
     * @return the partial `NDArray`
     */
    operator fun get(index: NDArray): NDArray = this[NDIndex("{}", index)]

    /**
     * Returns a partial `NDArray`.
     *
     * @param indices the indices used to indicate what to get
     * @param args arguments to replace the variable "{}" in the indices string. Can be an integer,
     * long, boolean [NDArray], or integer [NDArray].
     * @return the partial `NDArray`
     * @see NDIndex.NDIndex
     */
    operator fun get(indices: String, vararg args: Any): NDArray = this[NDIndex(indices, *args)]

    /**
     * Returns a partial `NDArray`.
     *
     * @param indices the indices with each index corresponding to the dimensions and negative
     * indices starting from the end
     * @return the partial `NDArray`
     */
    fun get(vararg indices: Long): NDArray = this[NDIndex(*indices)]

    /**
     * Returns a partial `NDArray`.
     *
     * @param manager the manager used to create the arrays
     * @param indices the indices with each index corresponding to the dimensions and negative
     * indices starting from the end
     * @return the partial `NDArray`
     */
    fun get(manager: NDManager, vararg indices: Long): NDArray = this[manager, NDIndex(*indices)]

    /**
     * Returns a partial `NDArray` pointed by the indexed array.
     *
     * <pre>
     * out[i][j][k] = input[index[i][j][k]][j][k] # if axis == 0
     * out[i][j][k] = input[i][index[i][j][k]][k] # if axis == 1
     * out[i][j][k] = input[i][j][index[i][j][k]] # if axis == 2
    </pre> *
     *
     * @param index picks the elements of an NDArray to the same position as index
     * @param axis the entries of index are indices of axis
     * @return the partial `NDArray` of the same shape as index
     */
    fun gather(index: NDArray, axis: Int): NDArray

    /**
     * Returns a partial `NDArray` pointed by the indexed array.
     *
     * <pre>
     * Given NDArray arr and NDArray idx. idx is the following structure:
     * \( idx = [ idx[0, ...], idx[1, ...],..., idx[indexingDepth,...] ] \)
     * corresponding to x, y, z index, i.e. [idx_x, idx_y, idx_z, ...].
    </pre> *
     *
     *
     * So indexingDepth smaller than or equal to data.shape[0] If indexingDepth is smaller than
     * data.shape[0], for instance, data.shape[0]=3, i.e. x,y,z but indexingDepth = 2, i.e. [idx_x,
     * idx_y], then the tail co-rank = data.shape[0] - indexingDepth will be kept.
     *
     *
     * With it, the output shape = idx_y.shape appended by data.shape[indexingDepth:] [mx.symbol.gather_nd](https://mxnet.apache.org/versions/1.6/api/r/docs/api/mx.symbol.gather_nd.html?highlight=gather_nd)
     *
     * @param index picks the elements of an NDArray to the same position as index
     * @return the partial `NDArray` of the same shape as index
     */
    infix fun gatherNd(index: NDArray): NDArray

    /**
     * Returns a partial `NDArray` pointed by index according to linear indexing, and the of
     * output is of the same shape as index.
     *
     * @param index picks the elements of an NDArray and output to the same entry as in index
     * @return the partial `NDArray` of the same shape as index
     */
    infix fun take(index: NDArray): NDArray = take(manager, index)

    /**
     * Returns a partial `NDArray` pointed by index according to linear indexing, and the of
     * output is of the same shape as index.
     *
     * @param manager the manager used to create the arrays
     * @param index picks the elements of an NDArray and output to the same entry as in index
     * @return the partial `NDArray` of the same shape as index
     */
    fun take(manager: NDManager, index: NDArray): NDArray

    /**
     * Sets the entries of `NDArray` pointed by index, according to linear indexing, to be the
     * numbers in value.
     *
     *
     * Value has to be of the same shape as index.
     *
     * @param index select the entries of an `NDArray`
     * @param value numbers to assign to the indexed entries
     * @return the NDArray with updated values
     */
    fun put(index: NDArray, value: NDArray): NDArray

    /**
     * Writes all values from the tensor value into self at the indices specified in the index
     * tensor.
     *
     * <pre>
     * This is the reverse operation of the manner described in gather().
     *
     * self[index[i][j][k]][j][k] = value[i][j][k] # if axis == 0
     * self[i][index[i][j][k]][k] = value[i][j][k] # if axis == 1
     * self[i][j][index[i][j][k]] = value[i][j][k] # if axis == 2
    </pre> *
     *
     * [torch.Tensor.scatter_](https://pytorch.org/docs/1.13/generated/torch.Tensor.scatter_.html#torch.Tensor.scatter)
     *
     * @param axis the axis along which to index
     * @param index the indices of elements to scatter, can be either empty or of the same
     * dimensionality as value. When empty, the operation returns self unchanged
     * @param value the source element(s) to scatter
     * @return the NDArray with updated values
     */
    fun scatter(index: NDArray, value: NDArray, axis: Int): NDArray

    /**
     * Returns a scalar `NDArray` corresponding to a single element.
     *
     * @param indices the indices of the scalar to return. Must return only a single element
     * @return a scalar `NDArray` corresponding to the element
     * @throws IllegalArgumentException thrown if the result is not a single element
     */
    fun getScalar(vararg indices: Long): NDArray {
        val value = get(NDIndex(*indices))
        require(value.size == 1L) { "The supplied Index does not produce a scalar" }
        return value
    }

    /**
     * Returns a long element from this `NDArray`.
     *
     * @param indices the indices of the long element to return
     * @return the element in the specified index as a long
     * @throws IllegalArgumentException thrown if the result is not a single element
     */
    fun getLong(vararg indices: Long): Long = getScalar(*indices).use { it.toLongArray()[0] }

    /**
     * Returns a double element from this `NDArray`.
     *
     * @param indices the indices of the double element to return
     * @return the element in the specified index as a double
     * @throws IllegalArgumentException thrown if the result is not a single element
     */
    fun getDouble(vararg indices: Long): Double = getScalar(*indices).use { it.toDoubleArray()[0] }

    /**
     * Returns a float element from this `NDArray`.
     *
     * @param indices the indices of the long element to return
     * @return the element in the specified index as a float
     * @throws IllegalArgumentException thrown if the result is not a single element
     */
    fun getFloat(vararg indices: Long): Float = getScalar(*indices).use { it.toFloatArray()[0] }

    /**
     * Returns an int element from this `NDArray`.
     *
     * @param indices the indices of the int element to return
     * @return the element in the specified index as an integer
     * @throws IllegalArgumentException thrown if the result is not a single element
     */
    fun getInt(vararg indices: Long): Int = getScalar(*indices).use { it.toIntArray()[0] }

    /**
     * Returns an byte element from this `NDArray`.
     *
     * @param indices the indices of the byte element to return
     * @return the element in the specified index as a byte
     * @throws IllegalArgumentException thrown if the result is not a single element
     */
    fun getByte(vararg indices: Long): Byte = getScalar(*indices).use { it.toByteArray()[0] }

    /**
     * Returns an integer element from this `NDArray` that represent unsigned integer with 8
     * bits.
     *
     * @param indices the indices of the unsigned 8 bits integer element to return
     * @return the element in the specified index as an uint8
     * @throws IllegalArgumentException thrown if the result is not a single element
     */
    fun getUint8(vararg indices: Long): UByte = getByte(*indices).toUByte()

    /**
     * Returns a boolean element from this `NDArray`.
     *
     * @param indices the indices of the int element to return
     * @return the element in the specified index as a boolean
     * @throws IllegalArgumentException thrown if the result is not a single element
     */
    fun getBoolean(vararg indices: Long): Boolean = getScalar(*indices).use { it.toBooleanArray()[0] }

    /**
     * Deep-copies the current `NDArray` to the one passed in.
     *
     * @param array this `NDArray` prepared to be copied to
     */
    infix fun copyTo(array: NDArray) = array.setFrom(toByteBuffer())

    /**
     * Returns a copy of this `NDArray`.
     *
     * @return a copy of this `NDArray`
     */
    fun duplicate(): NDArray {
        val array = manager.create(shape, dataType, device)
        array.name = name
        copyTo(array)
        return array
    }

    /**
     * Returns portion of this `NDArray` given the index boolean `NDArray` along first
     * axis.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {1f, 2f, 3f, 4f, 5f, 6f}, new Shape(3, 2));
     * jshell&gt; NDArray mask = manager.create(new boolean[] {true, false, true});
     * jshell&gt; array.booleanMask(mask);
     * ND: (2, 2) cpu() float32
     * [[1., 2.],
     * [5., 6.],
     * ]
    </pre> *
     *
     * @param index boolean `NDArray` mask
     * @return the result `NDArray`
     */
    infix fun booleanMask(index: NDArray): NDArray = booleanMask(index, 0)

    /**
     * Returns portion of this `NDArray` given the index boolean `NDArray` along given
     * axis.
     *
     * @param index boolean `NDArray` mask
     * @param axis an integer that represents the axis of `NDArray` to mask from
     * @return the result `NDArray`
     */
    fun booleanMask(index: NDArray, axis: Int): NDArray

    /**
     * Sets all elements outside the sequence to a constant value.
     *
     *
     * This function takes an n-dimensional input array of the form [batch_size,
     * max_sequence_length, ....] and returns an array of the same shape. Parameter `sequenceLength` is used to handle variable-length sequences. sequence_length should be an
     * input array of positive ints of dimension [batch_size].
     *
     * @param sequenceLength used to handle variable-length sequences
     * @param value the constant value to be set
     * @return the result `NDArray`
     */
    fun sequenceMask(sequenceLength: NDArray, value: Float): NDArray

    /**
     * Sets all elements outside the sequence to 0.
     *
     *
     * This function takes an n-dimensional input array of the form [batch_size,
     * max_sequence_length, ....] and returns an array of the same shape. Parameter `sequenceLength` is used to handle variable-length sequences. sequence_length should be an
     * input array of positive ints of dimension [batch_size].
     *
     * @param sequenceLength used to handle variable-length sequences
     * @return the result `NDArray`
     */
    fun sequenceMask(sequenceLength: NDArray): NDArray

    /**
     * Returns an `NDArray` of zeros with the same [Shape], [DataType] and [ ] as the input `NDArray`.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(6f).reshape(2, 3);
     * jshell&gt; array;
     * ND: (2, 3) cpu() float32
     * [[0., 1., 2.],
     * [3., 4., 5.],
     * ]
     * jshell&gt; array.zerosLike();
     * ND: (2, 3) cpu() float32
     * [[0., 0., 0.],
     * [0., 0., 0.],
     * ]
    </pre> *
     *
     * @return a `NDArray` filled with zeros
     */
    fun zerosLike(): NDArray = manager.zeros(shape, dataType, device)

    /**
     * Returns an `NDArray` of ones with the same [Shape], [DataType] and [ ] as the input `NDArray`.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(6f).reshape(2, 3);
     * jshell&gt; array;
     * ND: (2, 3) cpu() float32
     * [[0., 1., 2.],
     * [3., 4., 5.],
     * ]
     * jshell&gt; array.onesLike();
     * ND: (2, 3) cpu() float32
     * [[1., 1., 1.],
     * [1., 1., 1.],
     * ]
    </pre> *
     *
     * @return a `NDArray` filled with ones
     */
    fun onesLike(): NDArray = manager.ones(shape, dataType, device)

    /**
     * Returns an uninitialized `NDArray` with the same [Shape], [DataType] and
     * [SparseFormat] as the input `NDArray`.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(6f).reshape(2, 3);
     * jshell&gt; array;
     * ND: (2, 3) cpu() float32
     * [[0., 1., 2.],
     * [3., 4., 5.],
     * ]
     * jshell&gt; array.like(); // uninitialized NDArray
     * ND: (2, 3) cpu() float32
     * [[ 9.80908925e-45,  0.00000000e+00,  0.00000000e+00],
     * [ 0.00000000e+00,  7.61595174e-07,  2.80259693e-44],
     * ]
    </pre> *
     *
     * @return the result `NDArray`
     */
    fun like(): NDArray = manager.create(shape)

    ////////////////////////////////////////
    ////////////////////////////////////////
    // Operations
    ////////////////////////////////////////
    ////////////////////////////////////////
    ////////////////////////////////////////
    // Operations: Element Comparison
    ////////////////////////////////////////
    /**
     * Returns `true` if all elements in this `NDArray` are equal to the [Number].
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.ones(new Shape(2, 3));
     * jshell&gt; array.contentEquals(1); // return true instead of boolean NDArray
     * true
    </pre> *
     *
     * @param number the number to compare
     * @return the boolean result
     */
    infix fun contentEquals(number: Number): Boolean

    /**
     * Returns `true` if all elements in this `NDArray` are equal to the other [ ].
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array1 = manager.arange(6f).reshape(2, 3);
     * jshell&gt; NDArray array2 = manager.create(new float[] {0f, 1f, 2f, 3f, 4f, 5f}, new Shape(2, 3));
     * jshell&gt; array1.contentEquals(array2); // return true instead of boolean NDArray
     * true
    </pre> *
     *
     * @param other the other `NDArray` to compare
     * @return the boolean result
     */
    fun contentEquals(other: NDArray): Boolean

    /**
     * Checks 2 `NDArray`s for equal shapes.
     *
     *
     * Shapes are considered equal if:
     *
     *
     *  * Both `NDArray`s have equal rank, and
     *  * size(0)...size(rank()-1) are equal for both `NDArray`s
     *
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array1 = manager.ones(new Shape(1, 2, 3));
     * jshell&gt; NDArray array2 = manager.create(new Shape(1, 2, 3));
     * jshell&gt; array1.shapeEquals(array2); // return true instead of boolean NDArray
     * true
    </pre> *
     *
     * @param other the other `NDArray`
     * @return `true` if the [Shape]s are the same
     */
    infix fun shapeEquals(other: NDArray): Boolean = shape == other.shape

    /**
     * Returns `true` if two `NDArray`s are element-wise equal within a tolerance.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array1 = manager.create(new double[] {1e10, 1e-7});
     * jshell&gt; NDArray array2 = manager.create(new double[] {1.00001e10, 1e-8});
     * jshell&gt; array1.allClose(array2); // return false instead of boolean NDArray
     * false
     * jshell&gt; NDArray array1 = manager.create(new double[] {1e10, 1e-8});
     * jshell&gt; NDArray array2 = manager.create(new double[] {1.00001e10, 1e-9});
     * jshell&gt; array1.allClose(array2); // return true instead of boolean NDArray
     * true
    </pre> *
     *
     * @param other the `NDArray` to compare with
     * @return the boolean result
     */
    infix fun allClose(other: NDArray): Boolean = allClose(other, 1e-5, 1e-08, false)

    /**
     * Returns `true` if two `NDArray` are element-wise equal within a tolerance.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array1 = manager.create(new double[] {1e10, 1e-7});
     * jshell&gt; NDArray array2 = manager.create(new double[] {1.00001e10, 1e-8});
     * jshell&gt; array1.allClose(array2, 1e-05, 1e-08, false); // return false instead of boolean NDArray
     * false
     * jshell&gt; NDArray array1 = manager.create(new double[] {1e10, 1e-8});
     * jshell&gt; NDArray array2 = manager.create(new double[] {1.00001e10, 1e-9});
     * jshell&gt; array1.allClose(array2, 1e-05, 1e-08, false); // return true instead of boolean NDArray
     * true
     * jshell&gt; NDArray array1 = manager.create(new float[] {1f, Float.NaN});
     * jshell&gt; NDArray array2 = manager.create(new float[] {1f, Float.NaN});
     * jshell&gt; array1.allClose(array2, 1e-05, 1e-08, true); // return true instead of boolean NDArray
     * true
    </pre> *
     *
     * @param other the `NDArray` to compare with
     * @param rtol the relative tolerance parameter
     * @param atol the absolute tolerance parameter
     * @param equalNan whether to compare NaN’s as equal. If `true`, NaN’s in the [     ] will be considered equal to NaN’s in the other `NDArray`
     * @return the boolean result
     */
    fun allClose(other: NDArray, rtol: Double, atol: Double, equalNan: Boolean): Boolean {
        if (!shapeEquals(other))
            return false
        val actualDoubleArray = toArray()
        val expectedDoubleArray = other.toArray()
        for (i in actualDoubleArray.indices) {
            val a = actualDoubleArray[i].toDouble()
            val b = expectedDoubleArray[i].toDouble()
            // handle NaN
            if (equalNan && a.isNaN() && b.isNaN())
                continue
            if (a.isNaN() || b.isNaN() || (abs(a - b) > atol + rtol * abs(b)))
                return false
        }
        return true
    }

    /**
     * Returns the boolean `NDArray` for element-wise "Equals" comparison.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.ones(new Shape(1));
     * jshell&gt; array.eq(1);
     * ND: (1) cpu() boolean
     * [ true]
    </pre> *
     *
     * @param n the number to compare
     * @return the boolean `NDArray` for element-wise "Equals" comparison
     */
    infix fun eq(n: Number): NDArray

    /**
     * Returns the boolean `NDArray` for element-wise "Equals" comparison.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array1 = manager.create(new float[] {0f, 1f, 3f});
     * jshell&gt; NDArray array2 = manager.arange(3f);
     * jshell&gt; array1.eq(array2);
     * ND: (3) cpu() boolean
     * [ true,  true, false]
    </pre> *
     *
     * @param other the `NDArray` to compare
     * @return the boolean `NDArray` for element-wise "Equals" comparison
     */
    infix fun eq(other: NDArray): NDArray

    /**
     * Returns the boolean `NDArray` for element-wise "Not equals" comparison.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(4f).reshape(2, 2);
     * jshell&gt; array.neq(1);
     * ND: (2, 2) cpu() boolean
     * [[ true, false],
     * [ true,  true],
     * ]
    </pre> *
     *
     * @param n the number to compare
     * @return the boolean `NDArray` for element-wise "Not equals" comparison
     */
    infix fun neq(n: Number): NDArray

    /**
     * Returns the boolean `NDArray` for element-wise "Not equals" comparison.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array1 = manager.create(new float[] {1f, 2f});
     * jshell&gt; NDArray array2 = manager.create(new float[] {1f, 3f});
     * jshell&gt; array1.neq(array2);
     * ND: (2) cpu() boolean
     * [false,  true]
     * jshell&gt; NDArray array1 = manager.create(new float[] {1f, 2f});
     * jshell&gt; NDArray array2 = manager.create(new float[] {1f, 3f, 1f, 4f}, new Shape(2, 2));
     * jshell&gt; array1.neq(array2); // broadcasting
     * ND: (2, 2) cpu() boolean
     * [[false,  true],
     * [false,  true],
     * ]
    </pre> *
     *
     * @param other the `NDArray` to compare
     * @return the boolean `NDArray` for element-wise "Not equals" comparison
     */
    infix fun neq(other: NDArray): NDArray

    /**
     * Returns the boolean `NDArray` for element-wise "Greater" comparison.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {4f, 2f});
     * jshell&gt; array.gt(2f);
     * ND: (2) cpu() boolean
     * [ true, false]
    </pre> *
     *
     * @param n the number to compare
     * @return the boolean `NDArray` for element-wise "Greater" comparison
     */
    infix fun gt(n: Number): NDArray

    /**
     * Returns the boolean `NDArray` for element-wise "Greater Than" comparison.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array1 = manager.create(new float[] {4f, 2f});
     * jshell&gt; NDArray array2 = manager.create(new float[] {2f, 2f});
     * jshell&gt; array1.neq(array2);
     * ND: (2) cpu() boolean
     * [ true, false]
    </pre> *
     *
     * @param other the `NDArray` to compare
     * @return the boolean `NDArray` for element-wis "Greater Than" comparison
     */
    infix fun gt(other: NDArray): NDArray

    /**
     * Returns the boolean `NDArray` for element-wise "Greater or equals" comparison.
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {4f, 2f});
     * jshell&gt; array.gte(2f);
     * ND: (2) cpu() boolean
     * [ true, true]
    </pre> *
     *
     * @param n the number to compare
     * @return the boolean `NDArray` for element-wise "Greater or equals" comparison
     */
    infix fun gte(n: Number): NDArray

    /**
     * Returns the boolean `NDArray` for element-wise "Greater or equals" comparison.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array1 = manager.create(new float[] {4f, 2f});
     * jshell&gt; NDArray array2 = manager.create(new float[] {2f, 2f});
     * jshell&gt; array1.gte(array2);
     * ND: (2) cpu() boolean
     * [ true, true]
    </pre> *
     *
     * @param other the number to compare
     * @return the boolean `NDArray` for "Greater or equals" comparison
     */
    infix fun gte(other: NDArray): NDArray

    /**
     * Returns the boolean `NDArray` for element-wise "Less" comparison.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {1f, 2f});
     * jshell&gt; array.lt(2f);
     * ND: (2) cpu() boolean
     * [ true, false]
    </pre> *
     *
     * @param n the number to compare
     * @return the boolean `NDArray` for element-wise "Less" comparison
     */
    infix fun lt(n: Number): NDArray

    /**
     * Returns the boolean `NDArray` for element-wise "Less" comparison.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array1 = manager.create(new float[] {1f, 2f});
     * jshell&gt; NDArray array2 = manager.create(new float[] {2f, 2f});
     * jshell&gt; array1.lt(array2);
     * ND: (2) cpu() boolean
     * [ true, false]
    </pre> *
     *
     * @param other the `NDArray` to compare
     * @return the boolean `NDArray` for element-wise "Less" comparison
     */
    infix fun lt(other: NDArray): NDArray

    /**
     * Returns the boolean `NDArray` for element-wise "Less or equals" comparison.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {1f, 2f});
     * jshell&gt; array.lte(2f);
     * ND: (2) cpu() boolean
     * [ true, true]
    </pre> *
     *
     * @param n the number to compare
     * @return the boolean `NDArray` for element-wise "Less or equals" comparison
     */
    infix fun lte(n: Number): NDArray

    /**
     * Returns the boolean `NDArray` for element-wise "Less or equals" comparison.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array1 = manager.create(new float[] {1f, 2f});
     * jshell&gt; NDArray array2 = manager.create(new float[] {2f, 2f});
     * jshell&gt; array1.lte(array2);
     * ND: (2) cpu() boolean
     * [ true, true]
    </pre> *
     *
     * @param other the `NDArray` to compare
     * @return the boolean `NDArray` for element-wise "Less or equals" comparison
     */
    infix fun lte(other: NDArray): NDArray

    ////////////////////////////////////////
    // Operations: Element Arithmetic
    ////////////////////////////////////////
    /**
     * Adds a number to this `NDArray` element-wise.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {1f, 2f});
     * jshell&gt; array.add(2f);
     * ND: (2) cpu() float32
     * [3., 4.]
    </pre> *
     *
     * @param n the number to add
     * @return the result `NDArray`
     */
    operator fun plus(n: Number): NDArray

    /**
     * Adds other `NDArray`s to this `NDArray` element-wise.
     *
     *
     * The shapes of this `NDArray` and other `NDArray` must be broadcastable.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array1 = manager.arange(9f).reshape(3, 3);
     * jshell&gt; NDArray array2 = manager.arange(3f);
     * jshell&gt; array1.add(array2); // broadcasting
     * ND: (3, 3) cpu() float32
     * [[ 0.,  2.,  4.],
     * [ 3.,  5.,  7.],
     * [ 6.,  8., 10.],
     * ]
    </pre> *
     *
     * @param other the other `NDArray`s to add
     * @return the result `NDArray`
     * @throws IllegalArgumentException others arrays must have at least one element
     */
    operator fun plus(other: NDArray): NDArray

    /**
     * Subtracts a number from this `NDArray` element-wise.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {1f, 2f});
     * jshell&gt; array.sub(2f);
     * ND: (2) cpu() float32
     * [-1.,  0.]
    </pre> *
     *
     * @param n the number to subtract from
     * @return the result `NDArray`
     */
    fun minus(n: Number): NDArray

    /**
     * Subtracts the other `NDArray` from this `NDArray` element-wise.
     *
     *
     * The shapes of this `NDArray` and other `NDArray`s must be broadcastable.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array1 = manager.arange(9).reshape(3, 3);
     * jshell&gt; NDArray array2 = manager.arange(3);
     * jshell&gt; array1.sub(array2); // broadcasting
     * ND: (3, 3) cpu() float32
     * [[0., 0., 0.],
     * [3., 3., 3.],
     * [6., 6., 6.],
     * ]
    </pre> *
     *
     * @param other the other `NDArray` to subtract from
     * @return the result `NDArray`
     */
    operator fun minus(other: NDArray): NDArray

    /**
     * Multiplies this `NDArray` by a number element-wise.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {1f, 2f});
     * jshell&gt; array.mul(3f);
     * ND: (2) cpu() float32
     * [3., 6.]
    </pre> *
     *
     * @param n the number to multiply by
     * @return the result `NDArray`
     */
    operator fun times(n: Number): NDArray

    /**
     * Multiplies this `NDArray` by other `NDArray`s element-wise.
     *
     *
     * The shapes of this `NDArray` and other `NDArray` must be broadcastable.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array1 = manager.arange(9f).reshape(3, 3);
     * jshell&gt; NDArray array2 = manager.arange(3f);
     * jshell&gt; array1.mul(array2); // broadcasting
     * ND: (3, 3) cpu() float32
     * [[ 0.,  1.,  4.],
     * [ 0.,  4., 10.],
     * [ 0.,  7., 16.],
     * ]
    </pre> *
     *
     * @param other the other `NDArray`s to multiply by
     * @return the result `NDArray`
     * @throws IllegalArgumentException others arrays must have at least one element
     */
    operator fun times(other: NDArray): NDArray

    /**
     * Divides this `NDArray` by a number element-wise.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(5f);
     * jshell&gt; array.div(4f);
     * ND: (5) cpu() float32
     * [0.  , 0.25, 0.5 , 0.75, 1.  ]
    </pre> *
     *
     * @param n the number to divide by
     * @return the result `NDArray`
     */
    operator fun div(n: Number): NDArray

    /**
     * Divides this `NDArray` by the other `NDArray` element-wise.
     *
     *
     * The shapes of this `NDArray` and the other `NDArray` must be broadcastable.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array1 = manager.arange(9f).reshape(3, 3);
     * jshell&gt; NDArray array2 = manager.ones(new Shape(3)).mul(10);
     * jshell&gt; array1.div(array2); // broadcasting
     * ND: (3, 3) cpu() float32
     * [[0. , 0.1, 0.2],
     * [0.3, 0.4, 0.5],
     * [0.6, 0.7, 0.8],
     * ]
    </pre> *
     *
     * @param other the other `NDArray` to divide by
     * @return the result `NDArray`
     */
    operator fun div(other: NDArray): NDArray

    /**
     * Returns element-wise remainder of division.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(7f);
     * jshell&gt; array.mod(5f);
     * ND: (7) cpu() float32
     * [0., 1., 2., 3., 4., 0., 1.]
    </pre> *
     *
     * @param n the divisor number
     * @return the result `NDArray`
     */
    operator fun rem(n: Number): NDArray

    /**
     * Returns element-wise remainder of division.
     *
     *
     * The shapes of this `NDArray` and the other `NDArray` must be broadcastable.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array1 = manager.create(new float[] {4f, 7f});
     * jshell&gt; NDArray array2 = manager.create(new float[] {2f, 3f});
     * jshell&gt; array1.mod(array2);
     * ND: (2) cpu() float32
     * [0., 1.]
    </pre> *
     *
     * @param other the divisor `NDArray`
     * @return the result `NDArray`
     */
    operator fun rem(other: NDArray): NDArray

    /**
     * Takes the power of this `NDArray` with a number element-wise.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(5f);
     * jshell&gt; array.pow(4f);
     * ND: (6) cpu() float32
     * [  0.,   1.,   8.,  27.,  64., 125.]
    </pre> *
     *
     * @param n the number to take the power with
     * @return the result `NDArray`
     */
    infix fun pow(n: Number): NDArray

    /**
     * Takes the power of this `NDArray` with the other `NDArray` element-wise.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array1 = manager.arange(6f).reshape(3, 2);
     * jshell&gt; NDArray array2 = manager.create(new float[] {2f, 3f});
     * jshell&gt; array1.pow(array2); // broadcasting
     * ND: (3, 2) cpu() float32
     * [[  0.,   1.],
     * [  4.,  27.],
     * [ 16., 125.],
     * ]
    </pre> *
     *
     * @param other the other `NDArray` to take the power with
     * @return the result `NDArray`
     */
    infix fun pow(other: NDArray): NDArray

    /**
     * Computes this * log(other).
     *
     * @param other other the other `NDArray`
     * @return the result `NDArray`
     */
    infix fun xlogy(other: NDArray): NDArray

    /**
     * Adds a number to this `NDArray` element-wise in place.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {1f, 2f});
     * jshell&gt; array.addi(2f);
     * ND: (2) cpu() float32
     * [3., 4.]
     * jshell&gt; array;
     * ND: (2) cpu() float32
     * [3., 4.]
    </pre> *
     *
     * @param n the number to add
     * @return the result `NDArray`
     */
    operator fun plusAssign(n: Number)

    /**
     * Adds a number to this `NDArray` element-wise in place.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {1f, 2f});
     * jshell&gt; array.addi(2f);
     * ND: (2) cpu() float32
     * [3., 4.]
     * jshell&gt; array;
     * ND: (2) cpu() float32
     * [3., 4.]
    </pre> *
     *
     * @param n the number to add
     * @return the result `NDArray`
     */
    fun plusInP(n: Number): NDArray

    /**
     * Adds other `NDArray`s to this `NDArray` element-wise in place.
     *
     *
     * The shapes of this `NDArray` and other `NDArray`s must be broadcastable.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array1 = manager.create(new float[] {1f, 2f});
     * jshell&gt; NDArray array2 = manager.create(new float[] {3f, 4f});
     * jshell&gt; array1.addi(array2);
     * ND: (2) cpu() float32
     * [4., 6.]
     * jshell&gt; array;
     * ND: (2) cpu() float32
     * [4., 6.]
    </pre> *
     *
     * @param other the other `NDArray`s to add
     * @return the result `NDArray`
     */
    operator fun plusAssign(other: NDArray)

    /**
     * Adds other `NDArray`s to this `NDArray` element-wise in place.
     *
     *
     * The shapes of this `NDArray` and other `NDArray`s must be broadcastable.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array1 = manager.create(new float[] {1f, 2f});
     * jshell&gt; NDArray array2 = manager.create(new float[] {3f, 4f});
     * jshell&gt; array1.addi(array2);
     * ND: (2) cpu() float32
     * [4., 6.]
     * jshell&gt; array;
     * ND: (2) cpu() float32
     * [4., 6.]
    </pre> *
     *
     * @param other the other `NDArray`s to add
     * @return the result `NDArray`
     */
    fun plusInP(other: NDArray): NDArray

    /**
     * Subtracts a number from this `NDArray` element-wise in place.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {1f, 2f});
     * jshell&gt; array.subi(2f);
     * ND: (2) cpu() float32
     * [-1.,  0.]
     * jshell&gt; array;
     * ND: (2) cpu() float32
     * [-1.,  0.]
    </pre> *
     *
     * @param n the number to subtract
     * @return the result `NDArray`
     */
    operator fun minusAssign(n: Number)

    /**
     * Subtracts a number from this `NDArray` element-wise in place.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {1f, 2f});
     * jshell&gt; array.subi(2f);
     * ND: (2) cpu() float32
     * [-1.,  0.]
     * jshell&gt; array;
     * ND: (2) cpu() float32
     * [-1.,  0.]
    </pre> *
     *
     * @param n the number to subtract
     * @return the result `NDArray`
     */
    fun minusInP(n: Number): NDArray

    /**
     * Subtracts the other `NDArray` from this `NDArray` element-wise in place.
     *
     *
     * The shapes of this `NDArray` and other `NDArray`s must be broadcastable.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array1 = manager.arange(9f).reshape(3, 3);
     * jshell&gt; NDArray array2 = manager.arange(3f);
     * jshell&gt; array1.subi(array2); // broadcasting
     * ND: (3, 3) cpu() float32
     * [[0., 0., 0.],
     * [3., 3., 3.],
     * [6., 6., 6.],
     * ]
     * jshell&gt; array1;
     * [[0., 0., 0.],
     * [3., 3., 3.],
     * [6., 6., 6.],
     * ]
    </pre> *
     *
     * @param other the other `NDArray` to subtract from
     * @return the result `NDArray`
     */
    operator fun minusAssign(other: NDArray)

    /**
     * Subtracts the other `NDArray` from this `NDArray` element-wise in place.
     *
     *
     * The shapes of this `NDArray` and other `NDArray`s must be broadcastable.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array1 = manager.arange(9f).reshape(3, 3);
     * jshell&gt; NDArray array2 = manager.arange(3f);
     * jshell&gt; array1.subi(array2); // broadcasting
     * ND: (3, 3) cpu() float32
     * [[0., 0., 0.],
     * [3., 3., 3.],
     * [6., 6., 6.],
     * ]
     * jshell&gt; array1;
     * [[0., 0., 0.],
     * [3., 3., 3.],
     * [6., 6., 6.],
     * ]
    </pre> *
     *
     * @param other the other `NDArray` to subtract from
     * @return the result `NDArray`
     */
    fun minusInP(other: NDArray): NDArray

    /**
     * Multiplies this `NDArray` by a number element-wise in place.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {1f, 2f});
     * jshell&gt; array.muli(3f);
     * ND: (2) cpu() float32
     * [3., 6.]
     * jshell&gt; array;
     * ND: (2) cpu() float32
     * [3., 6.]
    </pre> *
     *
     * @param n the number to multiply by
     * @return the result `NDArray`
     */
    operator fun timesAssign(n: Number)

    /**
     * Multiplies this `NDArray` by a number element-wise in place.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {1f, 2f});
     * jshell&gt; array.muli(3f);
     * ND: (2) cpu() float32
     * [3., 6.]
     * jshell&gt; array;
     * ND: (2) cpu() float32
     * [3., 6.]
    </pre> *
     *
     * @param n the number to multiply by
     * @return the result `NDArray`
     */
    fun timesInP(n: Number): NDArray

    /**
     * Multiplies this `NDArray` by other `NDArray` element-wise in place.
     *
     *
     * The shapes of this `NDArray` and other `NDArray`s must be broadcastable.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array1 = manager.arange(9f).reshape(3, 3);
     * jshell&gt; NDArray array2 = manager.arange(3f);
     * jshell&gt; array1.muli(array2); // broadcasting
     * ND: (3, 3) cpu() float32
     * [[ 0.,  1.,  4.],
     * [ 0.,  4., 10.],
     * [ 0.,  7., 16.],
     * ]
     * jshell&gt; array1;
     * ND: (3, 3) cpu() float32
     * [[ 0.,  1.,  4.],
     * [ 0.,  4., 10.],
     * [ 0.,  7., 16.],
     * ]
    </pre> *
     *
     * @param other the other NDArrays to multiply with
     * @return the result `NDArray`
     */
    operator fun timesAssign(other: NDArray)

    /**
     * Multiplies this `NDArray` by other `NDArray` element-wise in place.
     *
     *
     * The shapes of this `NDArray` and other `NDArray`s must be broadcastable.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array1 = manager.arange(9f).reshape(3, 3);
     * jshell&gt; NDArray array2 = manager.arange(3f);
     * jshell&gt; array1.muli(array2); // broadcasting
     * ND: (3, 3) cpu() float32
     * [[ 0.,  1.,  4.],
     * [ 0.,  4., 10.],
     * [ 0.,  7., 16.],
     * ]
     * jshell&gt; array1;
     * ND: (3, 3) cpu() float32
     * [[ 0.,  1.,  4.],
     * [ 0.,  4., 10.],
     * [ 0.,  7., 16.],
     * ]
    </pre> *
     *
     * @param other the other NDArrays to multiply with
     * @return the result `NDArray`
     */
    fun timesInP(other: NDArray): NDArray

    /**
     * Divides this `NDArray` by a number element-wise in place.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(5f);
     * jshell&gt; array.divi(4f);
     * ND: (5) cpu() float32
     * [0.  , 0.25, 0.5 , 0.75, 1.  ]
     * jshell&gt; array;
     * ND: (5) cpu() float32
     * [0.  , 0.25, 0.5 , 0.75, 1.  ]
    </pre> *
     *
     * @param n the number to divide values by
     * @return the array after applying division operation
     */
    operator fun divAssign(n: Number)

    /**
     * Divides this `NDArray` by a number element-wise in place.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(5f);
     * jshell&gt; array.divi(4f);
     * ND: (5) cpu() float32
     * [0.  , 0.25, 0.5 , 0.75, 1.  ]
     * jshell&gt; array;
     * ND: (5) cpu() float32
     * [0.  , 0.25, 0.5 , 0.75, 1.  ]
    </pre> *
     *
     * @param n the number to divide values by
     * @return the array after applying division operation
     */
    fun divInP(n: Number): NDArray

    /**
     * Divides this `NDArray` by the other `NDArray` element-wise in place.
     *
     *
     * The shapes of this `NDArray` and the other `NDArray` must be broadcastable.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array1 = manager.arange(9f).reshape(3, 3);
     * jshell&gt; NDArray array2 = manager.ones(new Shape(3)).mul(10);
     * jshell&gt; array1.divi(array2); // broadcasting
     * ND: (3, 3) cpu() float32
     * [[0. , 0.1, 0.2],
     * [0.3, 0.4, 0.5],
     * [0.6, 0.7, 0.8],
     * ]
     * jshell&gt; array1;
     * [[0. , 0.1, 0.2],
     * [0.3, 0.4, 0.5],
     * [0.6, 0.7, 0.8],
     * ]
    </pre> *
     *
     * @param other the other `NDArray` to divide by
     * @return the result of the divide
     */
    operator fun divAssign(other: NDArray)

    /**
     * Divides this `NDArray` by the other `NDArray` element-wise in place.
     *
     *
     * The shapes of this `NDArray` and the other `NDArray` must be broadcastable.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array1 = manager.arange(9f).reshape(3, 3);
     * jshell&gt; NDArray array2 = manager.ones(new Shape(3)).mul(10);
     * jshell&gt; array1.divi(array2); // broadcasting
     * ND: (3, 3) cpu() float32
     * [[0. , 0.1, 0.2],
     * [0.3, 0.4, 0.5],
     * [0.6, 0.7, 0.8],
     * ]
     * jshell&gt; array1;
     * [[0. , 0.1, 0.2],
     * [0.3, 0.4, 0.5],
     * [0.6, 0.7, 0.8],
     * ]
    </pre> *
     *
     * @param other the other `NDArray` to divide by
     * @return the result of the divide
     */
    fun divInP(other: NDArray): NDArray

    /**
     * Returns element-wise remainder of division in place.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(7f);
     * jshell&gt; array.modi(5f);
     * ND: (7) cpu() float32
     * [0., 1., 2., 3., 4., 0., 1.]
     * jshell&gt; array;
     * ND: (7) cpu() float32
     * [0., 1., 2., 3., 4., 0., 1.]
    </pre> *
     *
     * @param n the divisor number
     * @return the result `NDArray`
     */
    operator fun remAssign(n: Number)

    /**
     * Returns element-wise remainder of division in place.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(7f);
     * jshell&gt; array.modi(5f);
     * ND: (7) cpu() float32
     * [0., 1., 2., 3., 4., 0., 1.]
     * jshell&gt; array;
     * ND: (7) cpu() float32
     * [0., 1., 2., 3., 4., 0., 1.]
    </pre> *
     *
     * @param n the divisor number
     * @return the result `NDArray`
     */
    fun remInP(n: Number): NDArray

    /**
     * Returns in place element-wise remainder of division in place.
     *
     *
     * The shapes of this `NDArray` and the other `NDArray` must be broadcastable.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array1 = manager.create(new float[] {4f, 7f});
     * jshell&gt; NDArray array2 = manager.create(new float[] {2f, 3f});
     * jshell&gt; array1.modi(array2);
     * ND: (2) cpu() float32
     * [0., 1.]
     * jshell&gt; array1;
     * ND: (2) cpu() float32
     * [0., 1.]
    </pre> *
     *
     * @param other the divisor `NDArray`
     * @return the result of the divide
     */
    operator fun remAssign(other: NDArray)

    /**
     * Returns in place element-wise remainder of division in place.
     *
     *
     * The shapes of this `NDArray` and the other `NDArray` must be broadcastable.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array1 = manager.create(new float[] {4f, 7f});
     * jshell&gt; NDArray array2 = manager.create(new float[] {2f, 3f});
     * jshell&gt; array1.modi(array2);
     * ND: (2) cpu() float32
     * [0., 1.]
     * jshell&gt; array1;
     * ND: (2) cpu() float32
     * [0., 1.]
    </pre> *
     *
     * @param other the divisor `NDArray`
     * @return the result of the divide
     */
    fun remInP(other: NDArray): NDArray

    /**
     * Takes the power of this `NDArray` with a number element-wise in place.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(5f);
     * jshell&gt; array.powi(4f);
     * ND: (6) cpu() float32
     * [  0.,   1.,   8.,  27.,  64., 125.]
     * jshell&gt; array;
     * ND: (6) cpu() float32
     * [  0.,   1.,   8.,  27.,  64., 125.]
    </pre> *
     *
     * @param n the number to raise the power to
     * @return the result `NDArray`
     */
    infix fun powInP(n: Number): NDArray

    /**
     * Takes the power of this `NDArray` with the other `NDArray` element-wise in place.
     *
     *
     * The shapes of this `NDArray` and the other `NDArray` must be broadcastable.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array1 = manager.arange(6f).reshape(3, 2);
     * jshell&gt; NDArray array2 = manager.create(new float[] {2f, 3f});
     * jshell&gt; array1.powi(array2); // broadcasting
     * ND: (3, 2) cpu() float32
     * [[  0.,   1.],
     * [  4.,  27.],
     * [ 16., 125.],
     * ]
     * jshell&gt; array1;
     * ND: (3, 2) cpu() float32
     * [[  0.,   1.],
     * [  4.,  27.],
     * [ 16., 125.],
     * ]
    </pre> *
     *
     * @param other the other `NDArray` to take the power with
     * @return the result `NDArray`
     */
    infix fun powInP(other: NDArray): NDArray

    /**
     * Returns the element-wise sign.
     *
     * @return the result `NDArray`
     */
    fun sign(): NDArray

    /**
     * Returns the element-wise sign in-place.
     *
     * @return the result `NDArray`
     */
    fun signInP(): NDArray

    /**
     * Returns the maximum of this `NDArray` and a number element-wise.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {2f, 3f, 4f});
     * jshell&gt; array.maximum(3f);
     * ND: (3) cpu() float32
     * [3., 3., 4.]
    </pre> *
     *
     * @param n the number to be compared
     * @return the maximum of this `NDArray` and a number element-wise
     */
    infix fun max(n: Number): NDArray

    /**
     * Returns the maximum of this `NDArray` and the other `NDArray` element-wise.
     *
     *
     * The shapes of this `NDArray` and the other `NDArray` must be broadcastable.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array1 = manager.create(new float[] {2f, 3f, 4f});
     * jshell&gt; NDArray array2 = manager.create(new float[] {1f, 5f, 2f});
     * jshell&gt; array1.maximum(array2);
     * ND: (3) cpu() float32
     * [2., 5., 4.]
     * jshell&gt; NDArray array1 = manager.eye(2);
     * jshell&gt; NDArray array2 = manager.create(new float[] {0.5f, 2f});
     * jshell&gt; array1.maximum(array2); // broadcasting
     * ND: (2, 2) cpu() float32
     * [[1. , 2. ],
     * [0.5, 2. ],
     * ]
    </pre> *
     *
     * @param other the `NDArray` to be compared
     * @return the maximum of this `NDArray` and the other `NDArray` element-wise
     */
    infix fun max(other: NDArray): NDArray

    /**
     * Returns the minimum of this `NDArray` and a number element-wise.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {2f, 3f, 4f});
     * jshell&gt; array.minimum(3f);
     * ND: (3) cpu() float32
     * [2., 3., 3.]
    </pre> *
     *
     * @param n the number to be compared
     * @return the minimum of this `NDArray` and a number element-wise
     */
    infix fun min(n: Number): NDArray

    /**
     * Returns the minimum of this `NDArray` and the other `NDArray` element-wise.
     *
     *
     * The shapes of this `NDArray` and the other `NDArray` must be broadcastable.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array1 = manager.create(new float[] {2f, 3f, 4f});
     * jshell&gt; NDArray array2 = manager.create(new float[] {1f, 5f, 2f});
     * jshell&gt; array1.minimum(array2);
     * ND: (3) cpu() float32
     * [1., 3., 2.]
     * jshell&gt; NDArray array1 = manager.eye(2);
     * jshell&gt; NDArray array2 = manager.create(new float[] {0.5f, 2f});
     * jshell&gt; array1.minimum(array2); // broadcasting
     * ND: (2, 2) cpu() float32
     * [[0.5, 0. ],
     * [0. , 1. ],
     * ]
    </pre> *
     *
     * @param other the `NDArray` to be compared
     * @return the minimum of this `NDArray` and the other `NDArray` element-wise
     */
    infix fun min(other: NDArray): NDArray

    ////////////////////////////////////////
    // Operations: Basic Numeric
    ////////////////////////////////////////
    /**
     * Returns the numerical negative `NDArray` element-wise.
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(5f);
     * jshell&gt; array.neg();
     * ND: (5) cpu() float32
     * [-0., -1., -2., -3., -4.]
    </pre> *
     *
     * @return the result `NDArray`
     */
    operator fun unaryMinus(): NDArray

    /**
     * Returns the numerical negative `NDArray` element-wise in place.
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(5f);
     * jshell&gt; array.negi();
     * jshell&gt; array;
     * ND: (5) cpu() float32
     * [-0., -1., -2., -3., -4.]
    </pre> *
     *
     * @return the result `NDArray`
     */
    fun unaryMinusInP(): NDArray?

    /**
     * Returns the absolute value of this `NDArray` element-wise.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {-1f, -2f});
     * jshell&gt; array.abs();
     * ND: (2) cpu() float32
     * [1., 2.]
    </pre> *
     *
     * @return the result `NDArray`
     */
    fun abs(): NDArray

    /**
     * Returns the square of this `NDArray` element-wise.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {2f, -3f});
     * jshell&gt; array.square();
     * ND: (2) cpu() float32
     * [4., 9.]
    </pre> *
     *
     * @return the result `NDArray`
     */
    fun square(): NDArray

    /**
     * Returns the square root of this `NDArray` element-wise.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {4f});
     * jshell&gt; array.sqrt();
     * ND: (1) cpu() float32
     * [2., ]
    </pre> *
     *
     * @return the result `NDArray`
     */
    fun sqrt(): NDArray

    /**
     * Returns the cube-root of this `NDArray` element-wise.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {1f, 8f, 27f});
     * jshell&gt; array.cbrt();
     * ND: (3) cpu() float32
     * [1., 2., 3.]
    </pre> *
     *
     * @return the result `NDArray`
     */
    fun cbrt(): NDArray

    /**
     * Returns the floor of this `NDArray` element-wise.
     *
     *
     * The floor of the scalar x is the largest integer i, such that i &lt;= x.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {-1.7f, -1.5f, -0.2f, 0.2f, 1.5f, 1.7f, 2.0f});
     * jshell&gt; array.floor();
     * ND: (7) cpu() float32
     * [-2., -2., -1.,  0.,  1.,  1.,  2.]
    </pre> *
     *
     * @return the result `NDArray`
     */
    fun floor(): NDArray

    /**
     * Returns the ceiling of this `NDArray` element-wise.
     *
     *
     * The ceil of the scalar x is the smallest integer i, such that i &gt;= x.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {-1.7f, -1.5f, -0.2f, 0.2f, 1.5f, 1.7f, 2.0f});
     * jshell&gt; array.ceil();
     * ND: (7) cpu() float32
     * [-1., -1., -0.,  1.,  2.,  2.,  2.]
    </pre> *
     *
     * @return the result `NDArray`
     */
    fun ceil(): NDArray

    /**
     * Returns the round of this `NDArray` element-wise.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {-1.7f, -1.5f, -0.2f, 0.2f, 1.5f, 1.7f, 2.0f});
     * jshell&gt; array.round();
     * ND: (7) cpu() float32
     * [-2., -2., -0.,  0.,  2.,  2.,  2.]
    </pre> *
     *
     * @return the result `NDArray`
     */
    fun round(): NDArray

    /**
     * Returns the truncated value of this `NDArray` element-wise.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {-1.7f, -1.5f, -0.2f, 0.2f, 1.5f, 1.7f, 2.0f});
     * jshell&gt; array.trunc();
     * ND: (7) cpu() float32
     * [-1., -1., -0.,  0.,  1.,  1.,  2.]
    </pre> *
     *
     * @return the result `NDArray`
     */
    fun trunc(): NDArray

    /**
     * Returns the exponential value of this `NDArray` element-wise.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {0f, 2.5f});
     * jshell&gt; array.exp();
     * ND: (2) cpu() float32
     * [ 1.    , 12.1825]
    </pre> *
     *
     * @return the result `NDArray`
     */
    fun exp(): NDArray

    /**
     * Return the log of the absolute value of the gamma function of this `NDArray`
     * element-wise.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {0.5f, 1f, 1.5f});
     * jshell&gt; array.gammaln();
     * ND: (2) cpu() float32
     * [ 0.5724,  0.0000, -0.1208]
    </pre> *
     *
     * @return the result `NDArray`
     */
    fun gammaln(): NDArray

    /**
     * Returns the natural logarithmic value of this `NDArray` element-wise.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {0f, 2.5f});
     * jshell&gt; array.log();
     * ND: (2) cpu() float32
     * [  -inf, 0.9163]
    </pre> *
     *
     * @return the result `NDArray`
     */
    fun log(): NDArray

    /**
     * Returns the base 10 logarithm of this `NDArray` element-wise.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {1000f, 1f, 150f});
     * jshell&gt; array.log10();
     * ND: (3) cpu() float32
     * [3.    , 0.    , 2.1761]
    </pre> *
     *
     * @return the result `NDArray`
     */
    fun log10(): NDArray

    /**
     * Returns the base 2 logarithm of this `NDArray` element-wise.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {8, 1f, 5f});
     * jshell&gt; array.log2();
     * ND: (3) cpu() float32
     * [3.    , 0.    , 2.3219]
    </pre> *
     *
     * @return the result `NDArray`
     */
    fun log2(): NDArray

    /**
     * Returns the trigonometric sine of this `NDArray` element-wise.
     *
     *
     * The input should be in radians (2 Pi radians equals 360 degrees).
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {0f, 30f, 45f, 60f, 90f});
     * jshell&gt; array = array.mul(Math.PI).div(180f);
     * jshell&gt; array.sin();
     * ND: (5) cpu() float32
     * [0.    , 0.5   , 0.7071, 0.866 , 1.    ]
    </pre> *
     *
     * @return the result `NDArray`
     */
    fun sin(): NDArray

    /**
     * Returns the trigonometric cosine of this `NDArray` element-wise.
     *
     *
     * The input should be in radians (2 Pi radians equals 360 degrees).
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new double[] {0, Math.PI/2, Math.PI});
     * jshell&gt; array.cos();
     * ND: (3) cpu() float64
     * [  1.0000000e+00,   6.1232340e-17,  -1.0000000e+00],
    </pre> *
     *
     * @return the result `NDArray`
     */
    fun cos(): NDArray

    /**
     * Returns the trigonometric tangent of this `NDArray` element-wise.
     *
     *
     * The input should be in radians (2 Pi radians equals 360 degrees).
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new double[] {-Math.PI, Math.PI/2, Math.PI});
     * jshell&gt; array.tan();
     * ND: (3) cpu() float64
     * [  1.2246468e-16,   1.6331239e+16,  -1.2246468e-16],
    </pre> *
     *
     * @return the result `NDArray`
     */
    fun tan(): NDArray

    /**
     * Returns the inverse trigonometric sine of this `NDArray` element-wise.
     *
     *
     * The input should be in the range [-1, 1]. The output is in the closed interval of [-Pi/2,
     * Pi/2].
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {1f, -1f, 0f});
     * jshell&gt; array.asin();
     * ND: (3) cpu() float64
     * [ 1.5708, -1.5708,  0.    ]
    </pre> *
     *
     * @return the result `NDArray`
     */
    fun asin(): NDArray

    /**
     * Returns the inverse trigonometric cosine of this `NDArray` element-wise.
     *
     *
     * The input should be in the range [-1, 1]. The output is in the closed interval of [-Pi/2,
     * Pi/2].
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {1f, -1f});
     * jshell&gt; array.acos();
     * ND: (2) cpu() float64
     * [0.    , 3.1416]
    </pre> *
     *
     * @return the result `NDArray`
     */
    fun acos(): NDArray

    /**
     * Returns the inverse trigonometric tangent of this `NDArray` element-wise.
     *
     *
     * The input should be in the range [-1, 1]. The output is in the closed interval of [-Pi/2,
     * Pi/2].
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {0f, 1f});
     * jshell&gt; array.atan();
     * ND: (2) cpu() float64
     * [0.    , 0.7854]
    </pre> *
     *
     * @return the result `NDArray`
     */
    fun atan(): NDArray

    /**
     * Returns the element-wise arc-tangent of this/other choosing the quadrant correctly.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray x = manager.create(new float[] {0f, 1f});
     * jshell&gt; NDArray y = manager.create(new float[] {0f, -6f});
     * jshell&gt; x.atan2(y);
     * ND: (2) cpu() float64
     * [0.    , 2.9764]
    </pre> *
     *
     * @param other The other `NDArray`
     * @return the result `NDArray`
     */
    infix fun atan2(other: NDArray): NDArray

    /**
     * Returns the hyperbolic sine of this `NDArray` element-wise.
     *
     *
     * sinh(x)=0.5*(exp(x) - exp(-x))
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new double[] {0, Math.PI});
     * jshell&gt; array.sinh();
     * ND: (2) cpu() float64
     * [ 0.    , 11.5487]
    </pre> *
     *
     * @return the result `NDArray`
     */
    fun sinh(): NDArray

    /**
     * Returns the hyperbolic cosine of this `NDArray` element-wise.
     *
     *
     * cosh(x)=0.5*(exp(x)+exp(-x))
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new double[] {0, Math.PI});
     * jshell&gt; array.cosh();
     * ND: (2) cpu() float64
     * [ 1.    , 11.592 ]
    </pre> *
     *
     * @return the result `NDArray`
     */
    fun cosh(): NDArray

    /**
     * Returns the hyperbolic tangent of this `NDArray` element-wise.
     *
     *
     * tanh(x)=sinh(x)/cosh(x)
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new double[] {0, Math.PI});
     * jshell&gt; array.tanh();
     * ND: (2) cpu() float64
     * [0.    , 0.9963]
    </pre> *
     *
     * @return the result `NDArray`
     */
    fun tanh(): NDArray

    /**
     * Returns the inverse hyperbolic sine of this `NDArray` element-wise.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new double[] {Math.E, 10});
     * jshell&gt; array.asinh();
     * ND: (2) cpu() float64
     * [1.7254, 2.9982]
    </pre> *
     *
     * @return the result `NDArray`
     */
    fun asinh(): NDArray

    /**
     * Returns the inverse hyperbolic cosine of this `NDArray` element-wise.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new double[] {Math.E, 10});
     * jshell&gt; array.acosh();
     * ND: (2) cpu() float64
     * [1.6575, 2.9932]
    </pre> *
     *
     * @return the result `NDArray`
     */
    fun acosh(): NDArray

    /**
     * Returns the inverse hyperbolic tangent of this `NDArray` element-wise.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new double[] {0, -0.5});
     * jshell&gt; array.atanh();
     * ND: (2) cpu() float64
     * [ 0.    , -0.5493]
    </pre> *
     *
     * @return the result `NDArray`
     */
    fun atanh(): NDArray

    /**
     * Converts this `NDArray` from radians to degrees element-wise.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(6f).mul(Math.PI / 3);
     * jshell&gt; array.toDegrees();
     * ND: (6) cpu() float32
     * [  0.,  60., 120., 180., 240., 300.]
    </pre> *
     *
     * @return the result `NDArray`
     */
    fun toDegrees(): NDArray

    /**
     * Converts this `NDArray` from degrees to radians element-wise.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(6f).mul(60);
     * jshell&gt; array.toRadians();
     * ND: (6) cpu() float32
     * [0.    , 1.0472, 2.0944, 3.1416, 4.1888, 5.236 ]
    </pre> *
     *
     * @return the result `NDArray`
     */
    fun toRadians(): NDArray

    ////////////////////////////////////////
    // Operations: Reduction
    ////////////////////////////////////////
    /**
     * Returns the maximum of this `NDArray`.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(4f).reshape(2,2);
     * jshell&gt; array;
     * ND: (2, 2) cpu() float32
     * [[0., 1.],
     * [2., 3.],
     * ]
     * jshell&gt; array.max(); // Maximum of the flattened array
     * ND: () cpu() float32
     * 3.
     * jshell&gt; array.max().getFloat() // Use getFloat() to get native float
     * 3.0
    </pre> *
     *
     * @return the maximum of this `NDArray`
     */
    fun max(): NDArray

    /**
     * Returns the maximum of this `NDArray` along given axes.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(4f).reshape(2,2);
     * jshell&gt; array;
     * ND: (2, 2) cpu() float32
     * [[0., 1.],
     * [2., 3.],
     * ]
     * jshell&gt; array.max(new int[]{0}); // Maximum along the first axis
     * ND: (2) cpu() float32
     * [2., 3.]
     * jshell&gt; array.max(new int[]{1}); // Maximum along the second axis
     * ND: (2) cpu() float32
     * [1., 3.]
    </pre> *
     *
     * @param axes the axes along which to operate
     * @return the maximum of this `NDArray` with the specified axes removed from the Shape
     * containing the max
     * @see NDArray.max
     */
    fun max(axes: IntArray): NDArray = max(axes, false)

    /**
     * Returns the maximum of this `NDArray` along given axes.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(4f).reshape(2,2);
     * jshell&gt; array;
     * ND: (2, 2) cpu() float32
     * [[0., 1.],
     * [2., 3.],
     * ]
     * jshell&gt; array.max(new int[]{0}, true); // Maximum along the first axis and keep dimension
     * ND: (1, 2) cpu() float32
     * [[2., 3.],
     * ]
     * jshell&gt; array.max(new int[]{1}, true); // Maximum along the second axis and keep dimension
     * ND: (2, 1) cpu() float32
     * [[1.],
     * [3.],
     * ]
    </pre> *
     *
     * @param axes the axes along which to operate
     * @param keepDims `true` to keep the specified axes as size 1 in the output array, `false` to squeeze the values out of the output array.
     * @return the maximum of this `NDArray`
     */
    fun max(axes: IntArray, keepDims: Boolean): NDArray

    /**
     * Returns the minimum of this `NDArray`.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(4f).reshape(2,2);
     * jshell&gt; array;
     * ND: (2, 2) cpu() float32
     * [[0., 1.],
     * [2., 3.],
     * ]
     * jshell&gt; array.min(); // Minimum of the flattened array
     * ND: () cpu() float32
     * 0.
     * jshell&gt; array.min().getFloat(); // Use getFloat() to get native float
     * 0.0
    </pre> *
     *
     * @return the minimum of this `NDArray`
     */
    fun min(): NDArray

    /**
     * Returns the minimum of this `NDArray` along given axes.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(4f).reshape(2,2);
     * jshell&gt; array;
     * ND: (2, 2) cpu() float32
     * [[0., 1.],
     * [2., 3.],
     * ]
     * jshell&gt; array.min(new int[]{0}); // Minimum along the first axis
     * ND: (2) cpu() float32
     * [0., 1.]
     * jshell&gt; array.min(new int[]{1}); // Minimum along the second axis
     * ND: (2) cpu() float32
     * [0., 2.]
    </pre> *
     *
     * @param axes the axes along which to operate
     * @return the minimum of this `NDArray` with the specified axes removed from the Shape
     * containing the min
     * @see NDArray.min
     */
    fun min(axes: IntArray): NDArray = min(axes, false)

    /**
     * Returns the minimum of this `NDArray` along given axes.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(4f).reshape(2,2);
     * jshell&gt; array
     * ND: (2, 2) cpu() float32
     * [[0., 1.],
     * [2., 3.],
     * ]
     * jshell&gt; array.min(new int[]{0}, true) // Minimum along the first axis and keep dimension
     * ND: (1, 2) cpu() float32
     * [[0., 1.],
     * ]
     * jshell&gt; array.min(new int[]{1}, true) // Minimum along the second axis and keep dimension
     * ND: (2, 1) cpu() float32
     * [[0.],
     * [2.],
     * ]
    </pre> *
     *
     * @param axes the axes along which to operate
     * @param keepDims `true` to keep the specified axes as size 1 in the output array, `false` to squeeze the values out of the output array
     * @return the minimum of this `NDArray`
     */
    fun min(axes: IntArray, keepDims: Boolean): NDArray

    /**
     * Returns the sum of this `NDArray`.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {0.5f, 1.5f});
     * jshell&gt; array.sum();
     * ND: () cpu() float32
     * 2.
     * jshell&gt; array.sum().getFloat(); // Use getFloat() to get native float
     * 2.0
     * jshell&gt; NDArray array = manager.create(new float[] {0f, 1f, 0f, 5f}, new Shape(2, 2));
     * jshell&gt; array.sum();
     * ND: () cpu() float32
     * 6.
    </pre> *
     *
     * @return the sum of this `NDArray`
     */
    fun sum(): NDArray

    /**
     * Returns the sum of this `NDArray` along given axes.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {0f, 1f, 0f, 5f}, new Shape(2, 2));
     * jshell&gt; array;
     * ND: (2, 2) cpu() float32
     * [[0., 1.],
     * [0., 5.],
     * ]
     * jshell&gt; array.sum(new int[] {0});
     * ND: (2) cpu() float32
     * [0., 6.]
     * jshell&gt; array.sum(new int[] {1});
     * ND: (2) cpu() float32
     * [1., 5.]
    </pre> *
     *
     * @param axes the axes along which to operate
     * @return the sum of this `NDArray` with the specified axes removed from the Shape
     * containing the sum
     * @see NDArray.sum
     */
    fun sum(axes: IntArray): NDArray = sum(axes, false)

    /**
     * Returns the sum of this `NDArray` along given axes.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {0f, 1f, 0f, 5f}, new Shape(2, 2));
     * jshell&gt; array;
     * ND: (2, 2) cpu() float32
     * [[0., 1.],
     * [0., 5.],
     * ]
     * jshell&gt; array.sum(new int[] {0}, true);
     * ND: (1, 2) cpu() float32
     * [[0., 6.],
     * ]
     * jshell&gt; array.sum(new int[] {1}, true);
     * ND: (2, 2) cpu() float32
     * [[0., 1.],
     * [0., 5.],
     * ]
    </pre> *
     *
     * @param axes the axes along which to operate
     * @param keepDims `true` to keep the specified axes as size 1 in the output array, `false` to squeeze the values out of the output array
     * @return the sum of this `NDArray`
     */
    fun sum(axes: IntArray, keepDims: Boolean): NDArray

    /**
     * Returns the cumulative product of elements of input in the dimension dim. For example, if
     * input is a vector of size N, the result will also be a vector of size N, with elements. [x1,
     * x1 * x2, x1 * x2 *x3 ...]
     *
     * @param axis the axis along which to operate
     * @return the cumulative product of this
     */
    infix fun cumProd(axis: Int): NDArray

    /**
     * Returns the cumulative product of elements of input in the dimension dim. For example, if
     * input is a vector of size N, the result will also be a vector of size N, with elements. [x1,
     * x1 * x2, x1 * x2 *x3 ...]
     *
     * @param axis the axis along which to operate
     * @param dataType the datatype of the output
     * @return the cumulative product of this
     */
    fun cumProd(axis: Int, dataType: DataType): NDArray

    /**
     * Returns the product of this `NDArray`.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {2f, 3f});
     * jshell&gt; array.prod();
     * ND: () cpu() float32
     * 6.
     * jshell&gt; array.prod().getFloat(); // Use getFloat to get native float
     * 6.0
     * jshell&gt; NDArray array = manager.create(new float[] {1f, 2f, 3f, 4f}, new Shape(2, 2));
     * jshell&gt; array.prod();
     * ND: () cpu() float32
     * 24.
    </pre> *
     *
     * @return the product of this `NDArray`
     */
    fun prod(): NDArray

    /**
     * Returns the product of this `NDArray` elements over the given axes.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {1f, 2f, 3f, 4f}, new Shape(2, 2));
     * jshell&gt; array;
     * ND: (2, 2) cpu() float32
     * [[1., 2.],
     * [3., 4.],
     * ]
     * jshell&gt; array.prod(new int[] {0});
     * ND: (2) cpu() float32
     * [3., 8.]
     * jshell&gt; array.prod(new int[] {1});
     * ND: (2) cpu() float32
     * [ 2., 12.]
    </pre> *
     *
     * @param axes the axes along which to operate
     * @return the product of this `NDArray` with the specified axes removed from the Shape
     * containing the prod
     * @see NDArray.prod
     */
    fun prod(axes: IntArray): NDArray = prod(axes, false)

    /**
     * Returns the product of this `NDArray` elements over the given axes.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {1f, 2f, 3f, 4f}, new Shape(2, 2));
     * jshell&gt; array;
     * ND: (2, 2) cpu() float32
     * [[1., 2.],
     * [3., 4.],
     * ]
     * jshell&gt; array.prod(new int[] {0}, true);
     * ND: (1, 2) cpu() float32
     * [[3., 8.],
     * ]
     * jshell&gt; array.prod(new int[] {1}, true);
     * ND: (2, 1) cpu() float32
     * [[ 2.],
     * [12.],
     * ]
    </pre> *
     *
     * @param axes the axes along which to operate
     * @param keepDims `true` to keep the specified axes as size 1 in the output array, `false` to squeeze the values out of the output array
     * @return the product of this `NDArray`
     */
    fun prod(axes: IntArray, keepDims: Boolean): NDArray

    /**
     * Returns the average of this `NDArray`.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {2f, 3f});
     * jshell&gt; array.mean();
     * ND: () cpu() float32
     * 2.5
     * jshell&gt; array.mean().getFloat(); // Use getFloat() to get native float
     * 2.5
     * jshell&gt; NDArray array = manager.create(new float[] {1f, 2f, 3f, 4f}, new Shape(2, 2));
     * jshell&gt; array.mean();
     * ND: () cpu() float32
     * 2.5
    </pre> *
     *
     * @return the average of this `NDArray`
     */
    fun mean(): NDArray

    /**
     * Returns the average of this `NDArray` along given axes.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {1f, 2f, 3f, 4f}, new Shape(2, 2));
     * jshell&gt; array;
     * ND: (2, 2) cpu() float32
     * [[1., 2.],
     * [3., 4.],
     * ]
     * jshell&gt; array.mean(new int[] {0});
     * ND: (2) cpu() float32
     * [2., 3.]
     * jshell&gt; array.mean(new int[] {1});
     * ND: (2) cpu() float32
     * [1.5, 3.5]
    </pre> *
     *
     * @param axes the axes along which to operate
     * @return the average of this `NDArray` with the specified axes removed from the Shape
     * containing the mean
     * @see NDArray.mean
     */
    fun mean(axes: IntArray): NDArray = mean(axes, false)

    /**
     * Returns the average of this `NDArray` along given axes.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {1f, 2f, 3f, 4f}, new Shape(2, 2));
     * jshell&gt; array;
     * ND: (2, 2) cpu() float32
     * [[1., 2.],
     * [3., 4.],
     * ]
     * jshell&gt; array.mean(new int[] {0}, true);
     * ND: (1, 2) cpu() float32
     * [[2., 3.],
     * ]
     * jshell&gt; array.mean(new int[] {1}, true);
     * ND: (2, 1) cpu() float32
     * [[1.5],
     * [3.5],
     * ]
    </pre> *
     *
     * @param axes the axes along which to operate
     * @param keepDims `true` to keep the specified axes as size 1 in the output array, `false` to squeeze the values out of the output array
     * @return the average of this `NDArray`
     */
    fun mean(axes: IntArray, keepDims: Boolean): NDArray

    /**
     * Performs Lp normalization of the array over specified dimension.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {1, 2, 3, 4, 5, 6}, new Shape(2, 3));
     * jshell&gt; array;
     * ND: (2, 2) cpu() float32
     * [[1., 2., 3.],
     * [4., 5., 6.],
     * ]
     * jshell&gt; array.normalize();
     * ND: (2, 3) cpu() float32
     * [[0.2673, 0.5345, 0.8018],
     * [0.4558, 0.5698, 0.6838],
     * ]
    </pre> *
     *
     * @return the normalized `NDArray`
     */
    fun normalize(): NDArray = normalize(2.0, 1, 1e-12)

    /**
     * Performs Lp normalization of the array over specified dimension.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {1, 2, 3, 4, 5, 6}, new Shape(2, 3));
     * jshell&gt; array;
     * ND: (2, 2) cpu() float32
     * [[1., 2., 3.],
     * [4., 5., 6.],
     * ]
     * jshell&gt; array.normalize(2, 1);
     * ND: (2, 3) cpu() float32
     * [[0.2673, 0.5345, 0.8018],
     * [0.4558, 0.5698, 0.6838],
     * ]
    </pre> *
     *
     * @param exponent the exponent value in the norm formulation
     * @param dim the dimension to reduce
     * @return the normalized `NDArray`
     */
    fun normalize(exponent: Double, dim: Long): NDArray = normalize(exponent, dim, 1e-12)

    /**
     * Performs Lp normalization of the array over specified dimension.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {1, 2, 3, 4, 5, 6}, new Shape(2, 3));
     * jshell&gt; array;
     * ND: (2, 2) cpu() float32
     * [[1., 2., 3.],
     * [4., 5., 6.],
     * ]
     * jshell&gt; array.normalize(2, 1, 1e-12);
     * ND: (2, 3) cpu() float32
     * [[0.2673, 0.5345, 0.8018],
     * [0.4558, 0.5698, 0.6838],
     * ]
    </pre> *
     *
     * @param exponent the exponent value in the norm formulation
     * @param dim the dimension to reduce
     * @param eps the small value to avoid division by zero
     * @return the normalized `NDArray`
     */
    fun normalize(exponent: Double, dim: Long, eps: Double): NDArray

    /**
     * Rotates an array by 90 degrees in the plane specified by axes.
     *
     *
     * Rotation direction is from the first towards the second axis.
     *
     * @param times Number of times the array is rotated by 90 degrees.
     * @param axes The array is rotated in the plane defined by the axes. Axes must be different.
     * @return the rotated NDArray
     */
    fun rotate90(times: Int, axes: IntArray): NDArray

    /**
     * Returns the sum along diagonals of this `NDArray`.
     *
     *
     * If this `NDArray` is 2-D, the sum along its diagonal is returned. If the [ ] has more than two dimensions, then the axes specified by axis1 and axis2 are used to
     * determine the 2-D sub-arrays whose traces are returned. The [Shape] of the resulting
     * [NDArray] is the same as that of a with axis1 and axis2 removed.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.eye(3);
     * jshell&gt; array;
     * ND: (3, 3) cpu() float32
     * [[1., 0., 0.],
     * [0., 1., 0.],
     * [0., 0., 1.],
     * ]
     * jshell&gt; array.trace();
     * ND: () cpu() float32
     * 3.
     * jshell&gt; NDArray array = manager.arange(8f).reshape(2, 2, 2);
     * jshell&gt; array;
     * ND: (2, 2, 2) cpu() float32
     * [[[0., 1.],
     * [2., 3.],
     * ],
     * [[4., 5.],
     * [6., 7.],
     * ],
     * ]
     * jshell&gt; array.trace();
     * ND: (2) cpu() float32
     * [6., 8.]
    </pre> *
     *
     * @return the sum along diagonals of this `NDArray`
     */
    fun trace(): NDArray = trace(0, 0, 1)

    /**
     * Returns the sum along diagonals of this `NDArray`.
     *
     *
     * If this `NDArray` is 2-D, the sum along its diagonal with the given offset is
     * returned, i.e., the sum of elements a[i,i+offset] for all i. If this `NDArray` has more
     * than two dimensions, then the axes specified by axis1 and axis2 are used to determine the 2-D
     * sub-arrays whose traces are returned. The [Shape] of the resulting array is the same as
     * this `NDArray` with axis1 and axis2 removed.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.eye(3);
     * jshell&gt; array;
     * ND: (3, 3) cpu() float32
     * [[1., 0., 0.],
     * [0., 1., 0.],
     * [0., 0., 1.],
     * ]
     * jshell&gt; array.trace(1);
     * ND: () cpu() float32
     * 0.
     * jshell&gt; NDArray array = manager.arange(8f).reshape(2, 2, 2);
     * jshell&gt; array;
     * ND: (2, 2, 2) cpu() float32
     * [[[0., 1.],
     * [2., 3.],
     * ],
     * [[4., 5.],
     * [6., 7.],
     * ],
     * ]
     * jshell&gt; array.trace(1);
     * ND: (2) cpu() float32
     * [2., 3.]
    </pre> *
     *
     * @param offset offset of the diagonal from the main diagonal. Can be both positive and
     * negative.
     * @return the sum along diagonals of this `NDArray`
     */
    fun trace(offset: Int): NDArray = trace(offset, 0, 1)

    /**
     * Returns the sum along diagonals of this `NDArray`.
     *
     *
     * If this `NDArray` is 2-D, the sum along its diagonal with the given offset is
     * returned, i.e., the sum of elements a[i,i+offset] for all i. If this `NDArray` has more
     * than two dimensions, then the axes specified by axis1 and axis2 are used to determine the 2-D
     * sub-arrays whose traces are returned. The [Shape] of the resulting array is the same as
     * this `NDArray` with axis1 and axis2 removed.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(8f).reshape(2, 2, 2);
     * jshell&gt; array;
     * ND: (2, 2, 2) cpu() float32
     * [[[0., 1.],
     * [2., 3.],
     * ],
     * [[4., 5.],
     * [6., 7.],
     * ],
     * ]
     * jshell&gt; array.trace(1,1,2);
     * ND: (2) cpu() float32
     * [1., 5.]
    </pre> *
     *
     * @param offset offset of the diagonal from the main diagonal. Can be both positive and
     * negative.
     * @param axis1 axes to be used as the first axis of the 2-D sub-arrays from which the diagonals
     * should be taken
     * @param axis2 axes to be used as the second axis of the 2-D sub-arrays from which the
     * diagonals should be taken
     * @return the sum along diagonals of this `NDArray`
     */
    fun trace(offset: Int, axis1: Int, axis2: Int): NDArray

    ////////////////////////////////////////
    // Operations: Shapes and Arrays Manipulation
    ////////////////////////////////////////
    /**
     * Splits this `NDArray` into multiple sub`NDArray`s given sections along first
     * axis.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(9f);
     * jshell&gt; array.split(3).forEach(System.out::println);
     * ND: (3) cpu() float32
     * [0., 1., 2.]
     *
     * ND: (3) cpu() float32
     * [3., 4., 5.]
     *
     * ND: (3) cpu() float32
     * [6., 7., 8.]
    </pre> *
     *
     * @param sections this `NDArray` will be divided into N (sections) equal `NDArray`
     * @return an [NDList] with size(axis) `NDArray`s with [Shape] `this.shape.remove(axis) `
     * @see NDArray.split
     */
    fun split(sections: Long): NDList = split(sections, 0)

    /**
     * Splits this `NDArray` into multiple sub-`NDArray`s given indices along first
     * axis.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(8f);
     * jshell&gt; array.split(new int[] {3, 5, 6}).forEach(System.out::println);
     * ND: (3) cpu() float32
     * [0., 1., 2.]
     *
     * ND: (2) cpu() float32
     * [3., 4.]
     *
     * ND: (1) cpu() float32
     * [5.]
     *
     * ND: (2) cpu() float32
     * [6., 7.]
    </pre> *
     *
     * @param indices the entries indicate where along axis this `NDArray` is split. If an
     * index exceeds the dimension of this `NDArray` along axis, an empty sub-[     ] is returned correspondingly.
     * @return an NDList with size(axis) `NDArray`s with [Shape] `this.shape.remove(axis) `
     * @see NDArray.split
     */
    fun split(indices: LongArray): NDList = split(indices, 0)

    /**
     * Splits this `NDArray` into multiple sub`NDArray`s given sections along the given
     * axis.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(18f).reshape(2, 9);
     * jshell&gt; array;
     * ND: (2, 9) cpu() float32
     * [[ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.],
     * [ 9., 10., 11., 12., 13., 14., 15., 16., 17.],
     * ]
     * jshell&gt; array.split(3, 1).forEach(System.out::println);
     * ND: (2, 3) cpu() float32
     * [[ 0.,  1.,  2.],
     * [ 9., 10., 11.],
     * ]
     *
     * ND: (2, 3) cpu() float32
     * [[ 3.,  4.,  5.],
     * [12., 13., 14.],
     * ]
     *
     * ND: (2, 3) cpu() float32
     * [[ 6.,  7.,  8.],
     * [15., 16., 17.],
     * ]
    </pre> *
     *
     * @param sections this `NDArray` will be divided into N (sections) equal arrays along
     * axis
     * @param axis the axis to split along
     * @return an [NDList] with numOutputs `NDArray`s with [Shape] `(this.shape.axis /= axis) `
     * @throws IllegalArgumentException thrown if the numOutputs does not equally divide the given
     * axis
     */
    fun split(sections: Long, axis: Int): NDList {
        val axisSize = shape.shape[axis]
        require(axisSize % sections == 0L) { "array split does not result in an equal division" }
        val sectionSize = axisSize / sections
        val indices = LongArray(sections.toInt()) { it * sectionSize }
        return split(indices, axis)
    }

    /**
     * Splits this `NDArray` into multiple sub-`NDArray`s given indices along given
     * axis.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(18f).reshape(2, 9);
     * jshell&gt; array;
     * ND: (2, 9) cpu() float32
     * [[ 0.,  1.,  2.,  3.,  4.,  5.,  6.,  7.,  8.],
     * [ 9., 10., 11., 12., 13., 14., 15., 16., 17.],
     * ]
     * jshell&gt; array.split(new int[] {2,4,5}, 1).forEach(System.out::println);
     * ND: (2, 2) cpu() float32
     * [[ 0.,  1.],
     * [ 9., 10.],
     * ]
     *
     * ND: (2, 2) cpu() float32
     * [[ 2.,  3.],
     * [11., 12.],
     * ]
     *
     * ND: (2, 1) cpu() float32
     * [[ 4.],
     * [13.],
     * ]
     *
     * ND: (2, 4) cpu() float32
     * [[ 5.,  6.,  7.,  8.],
     * [14., 15., 16., 17.],
     * ]
    </pre> *
     *
     * @param indices the entries indicate where along axis this `NDArray` is split. If an
     * index exceeds the dimension of this `NDArray` along axis, an empty sub-array is
     * returned correspondingly
     * @param axis the axis to split along
     * @return an [NDList] with numOutputs `NDArray`s with [Shape] `(this.shape.axis /= axis) `
     */
    fun split(indices: LongArray, axis: Int): NDList

    /**
     * Flattens this `NDArray` into a 1-D `NDArray` in row-major order.
     *
     *
     * To flatten in column-major order, first transpose this `NDArray`
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[]{1f, 2f, 3f, 4f}, new Shape(2, 2));
     * jshell&gt; array.flatten();
     * ND: (4) cpu() float32
     * [1., 2., 3., 4.]
    </pre> *
     *
     * @return a 1-D `NDArray` of equal size
     */
    fun flatten(): NDArray

    /**
     * Flattens this `NDArray` into a partially flatten `NDArray`.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[]{1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f}, new Shape(2, 2, 2));
     * jshell&gt; array.flatten(0, 1);
     * ND: (4) cpu() float32
     * [[1., 2], [3., 4.], [5., 6.], [7., 8.]]
    </pre> *
     *
     * @param startDim the first dim to flatten, inclusive
     * @param endDim the last dim to flatten, inclusive
     * @return a partially fallen `NDArray`
     */
    fun flatten(startDim: Int, endDim: Int): NDArray

    /**
     * Computes the one-dimensional discrete Fourier Transform.
     *
     * @param length Length of the transformed axis of the output.
     * @return The truncated or zero-padded input, transformed along the axis indicated by axis, or
     * the last one if axis is not specified.
     */
    infix fun fft(length: Long): NDArray = fft(length, -1)

    /**
     * Computes the one-dimensional discrete Fourier Transform.
     *
     * @param length Length of the transformed axis of the output.
     * @param axis Axis over which to compute the FFT.
     * @return The truncated or zero-padded input, transformed along the axis indicated by axis, or
     * the last one if axis is not specified.
     */
    fun fft(length: Long, axis: Long): NDArray

    /**
     * Computes the Short Time Fourier Transform (STFT).
     *
     * @param nFft size of Fourier transform
     * @param hopLength the distance between neighboring sliding window frames. Default can be:
     * floor(n_fft / 4)
     * @param center whether to pad input on both sides.
     * @param window Desired window to use. Recommend for HanningWindow
     * @param returnComplex whether to return a complex tensor, or a real tensor with an extra last
     * dimension for the real and imaginary components.
     * @return A NDArray containing the STFT result with shape described above
     */
    fun stft(nFft: Long, hopLength: Long, center: Boolean, window: NDArray, returnComplex: Boolean): NDArray =
        stft(nFft, hopLength, center, window, false, returnComplex)

    /**
     * Computes the Short Time Fourier Transform (STFT).
     *
     * @param nFft size of Fourier transform
     * @param hopLength the distance between neighboring sliding window frames. Default can be:
     * floor(n_fft / 4)
     * @param center whether to pad input on both sides.
     * @param window Desired window to use. Recommend for HanningWindow
     * @param normalize controls whether to return the normalized STFT results
     * @param returnComplex whether to return a complex tensor, or a real tensor with an extra last
     * dimension for the real and imaginary components.
     * @return A NDArray containing the STFT result with shape described above
     */
    fun stft(nFft: Long,
             hopLength: Long,
             center: Boolean,
             window: NDArray,
             normalize: Boolean,
             returnComplex: Boolean): NDArray

    /**
     * Computes the two-dimensional Discrete Fourier Transform.
     *
     * @param sizes Sizes of the transformed axes of the output. Will be zero-padded or trimmed to
     * this size.
     * @param axes Axes over which to compute the 2D-FFT.
     * @return The truncated or zero-padded input, transformed along the axes.
     */
    fun fft2(sizes: LongArray, axes: LongArray): NDArray

    /**
     * Computes the two-dimensional Discrete Fourier Transform along the last 2 axes.
     *
     * @param sizes Sizes of the transformed axes of the output. Will be zero-padded or trimmed to
     * this size.
     * @return The truncated or zero-padded input, transformed along the last two axes
     */
    infix fun fft2(sizes: LongArray): NDArray = fft2(sizes, longArrayOf(-2, -1))

    /**
     * Computes the two-dimensional inverse Discrete Fourier Transform.
     *
     * @param sizes Sizes of the transformed axes of the output. Will be zero-padded or trimmed to
     * this size.
     * @param axes Axes over which to compute the 2D-Inverse-FFT.
     * @return The truncated or zero-padded input, transformed along the axes.
     */
    fun ifft2(sizes: LongArray, axes: LongArray): NDArray

    /**
     * Computes the two-dimensional inverse Discrete Fourier Transform along the last 2 axes.
     *
     * @param sizes Sizes of the transformed axes of the output. Will be zero-padded or trimmed to
     * this size.
     * @return The truncated or zero-padded input, transformed along the axes.
     */
    infix fun ifft2(sizes: LongArray): NDArray = ifft2(sizes, longArrayOf(-2, -1))

    /**
     * Reshapes this `NDArray` to the given [Shape].
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(6f);
     * jshell&gt; array;
     * ND: (6) cpu() float32
     * [0., 1., 2., 3., 4., 5.]
     * jshell&gt; array.reshape(2, 3);
     * ND: (2, 3) cpu() float32
     * [[0., 1., 2.],
     * [3., 4., 5.],
     * ]
    </pre> *
     *
     * @param newShape the long array to reshape into. Must have equal size to the current shape
     * @return a reshaped `NDArray`
     * @throws IllegalArgumentException thrown if the given [Shape] does not match the size of
     * the current shape
     */
    fun reshape(vararg newShape: Long): NDArray = reshape(Shape(*newShape))

    /**
     * Reshapes this `NDArray` to the given [Shape].
     *
     *
     * You can reshape it to match another NDArray by calling `a.reshape(b.getShape()) `
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(6f);
     * jshell&gt; array;
     * ND: (6) cpu() float32
     * [0., 1., 2., 3., 4., 5.]
     * jshell&gt; array.reshape(new Shape(2, 3));
     * ND: (2, 3) cpu() float32
     * [[0., 1., 2.],
     * [3., 4., 5.],
     * ]
    </pre> *
     *
     * @param shape the [Shape] to reshape into. Must have equal size to the current shape
     * @return a reshaped `NDArray`
     * @throws IllegalArgumentException thrown if the given [Shape] does not match the size of
     * the current shape
     */
    infix fun reshape(shape: Shape): NDArray

    /**
     * Expands the [Shape] of a `NDArray`.
     *
     *
     * Inserts a new axis that will appear at the axis position in the expanded `NDArray`
     * shape.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {1f, 2f});
     * jshell&gt; array;
     * ND: (2) cpu() float32
     * [1., 2.]
     * jshell&gt; array.expandDims(0);
     * ND: (1, 2) cpu() float32
     * [[1., 2.],
     * ]
     * jshell&gt; array.expandDims(1);
     * ND: (2, 1) cpu() float32
     * [[1.],
     * [2.],
     * ]
    </pre> *
     *
     * @param axis the position in the expanded axes where the new axis is placed
     * @return the result `NDArray`. The number of dimensions is one greater than that of the
     * `NDArray`
     */
    infix fun expandDims(axis: Int): NDArray

    /**
     * Removes all singleton dimensions from this `NDArray` [Shape].
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {0f, 1f, 2f}, new Shape(1, 3, 1));
     * jshell&gt; array;
     * ND: (1, 3, 1) cpu() float32
     * [[[0.],
     * [1.],
     * [2.],
     * ],
     * ]
     * jshell&gt; array.squeeze();
     * ND: (3) cpu() float32
     * [0., 1., 2.]
    </pre> *
     *
     * @return a result `NDArray` of same size and data without singleton dimensions
     */
    fun squeeze(): NDArray {
        val shape = shape.shape
        return squeeze((0..shape.size).filter { shape[it] == 1L }.toIntArray())
    }

    /**
     * Removes a singleton dimension at the given axis.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {0f, 1f, 2f}, new Shape(1, 3, 1));
     * jshell&gt; array;
     * ND: (1, 3, 1) cpu() float32
     * [[[0.],
     * [1.],
     * [2.],
     * ],
     * ]
     * jshell&gt; array.squeeze(0);
     * ND: (3, 1) cpu() float32
     * [[0.],
     * [1.],
     * [2.],
     * ]
     * jshell&gt; array.squeeze(2);
     * ND: (1, 3) cpu() float32
     * [[0., 1., 2.],
     * ]
    </pre> *
     *
     * @param axis the axis at which to remove the singleton dimension
     * @return a result `NDArray` of same size and data without the axis at part of the shape
     * @throws IllegalArgumentException thrown if the given axis is not a singleton dimension
     */
    infix fun squeeze(axis: Int): NDArray = squeeze(intArrayOf(axis))

    /**
     * Removes singleton dimensions at the given axes.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {0f, 1f, 2f}, new Shape(1, 3, 1));
     * jshell&gt; array;
     * ND: (1, 3, 1) cpu() float32
     * [[[0.],
     * [1.],
     * [2.],
     * ],
     * ]
     * jshell&gt; array.squeeze(new int[] {0, 2});
     * ND: (3) cpu() float32
     * [0., 1., 2.]
    </pre> *
     *
     * @param axes the axes at which to remove the singleton dimensions
     * @return a result `NDArray` of same size and data without the axes at part of the shape
     * @throws IllegalArgumentException thrown if any of the given axes are not a singleton
     * dimension
     */
    infix fun squeeze(axes: IntArray): NDArray

    /**
     * Returns the unique elements of the input tensor.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {3f, 1f, 2f, 3f, 1f, 2f, 1f, 3f, 2f}, new Shape(3, 3));
     * jshell&gt; array;
     * ND: (3, 3) cpu() float32
     * [[[3., 1., 2.],
     * [3., 1., 2.],
     * [1., 3., 2.],
     * ],
     * ]
     * jshell&gt; NDList output = array.unique(0, true, true, true);
     * jshell&gt; output.get(0);
     * jshell&gt; output.get(1);
     * jshell&gt; output.get(2);
     *
     * ND: (2, 3) cpu() float32
     * [[1., 3., 2.],
     * [3., 1., 2.],
     * ]
     *
     * ND: (3) cpu() int64
     * [ 1,  1,  0]
     *
     * ND: (2) cpu() int64
     * [ 1,  2]
     *
    </pre> *
     *
     * @param dim the dimension to apply unique
     * @param sorted whether to sort the unique elements in ascending order before returning as
     * output
     * @param returnInverse return the indices which, fed into the output unique array as indices,
     * will recover the original array
     * @param returnCounts return the counts for each unique element
     * @return An `NDList` containing: output (Tensor): the output list of unique elements or
     * low-rank tensors. inverse_indices (Tensor): (optional) if return_inverse is True, there
     * will be an additional returned tensor (same shape as input) representing the indices for
     * where elements in the original input map to in the output; otherwise, this function will
     * only return a single tensor. counts (Tensor): (optional) if return_counts is True, there
     * will be an additional returned tensor (same shape as output or output.size(dim), if dim
     * was specified) representing the number of occurrences for each unique value or tensor.
     */
    fun unique(dim: Int?, sorted: Boolean, returnInverse: Boolean, returnCounts: Boolean): NDList

    /**
     * Returns the unique elements of the input tensor. The output is flattened.
     *
     * @param sorted whether to sort the unique elements in ascending order before returning as
     * output
     * @param returnInverse return the indices which, fed into the output unique array as indices,
     * will recover the original array
     * @param returnCounts return the counts for each unique element
     * @return An `NDList` containing: output (Tensor): the output list of unique elements or
     * low-rank tensors. inverse_indices (Tensor): (optional) if return_inverse is True, there
     * will be an additional returned tensor (same shape as input) representing the indices for
     * where elements in the original input map to in the output; otherwise, this function will
     * only return a single tensor. counts (Tensor): (optional) if return_counts is True, there
     * will be an additional returned tensor (same shape as output or output.size(dim), if dim
     * was specified) representing the number of occurrences for each unique value or tensor.
     */
    fun unique(sorted: Boolean, returnInverse: Boolean, returnCounts: Boolean): NDList =
        unique(null, sorted, returnInverse, returnCounts)

    /**
     * Returns the unique elements of the input tensor. The output is flattened.
     *
     * @return An `NDList` containing: output (Tensor): the output list of unique elements or
     * low-rank tensors. inverse_indices (Tensor): (optional) if return_inverse is True, there
     * will be an additional returned tensor (same shape as input) representing the indices for
     * where elements in the original input map to in the output; otherwise, this function will
     * only return a single tensor. counts (Tensor): (optional) if return_counts is True, there
     * will be an additional returned tensor (same shape as output or output.size(dim), if dim
     * was specified) representing the number of occurrences for each unique value or tensor.
     */
    fun unique(): NDList = unique(null, sorted = true, returnInverse = false, returnCounts = false)

    /**
     * Joins a `NDArray` along a new axis.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array1 = manager.create(new float[] {0f, 1f});
     * jshell&gt; NDArray array2 = manager.create(new float[] {2f, 3f});
     * jshell&gt; array1.stack(array2, 0);
     * ND: (2, 2) cpu() float32
     * [[0., 1.],
     * [2., 3.],
     * ]
     * jshell&gt; array1.stack(array2, 1);
     * ND: (2, 2) cpu() float32
     * [[0., 2.],
     * [1., 3.],
     * ]
    </pre> *
     *
     * @param array the input `NDArray` which must have the same [Shape]as this `NDArray`
     * @param axis the axis in the result `NDArray` along which the input `NDArray` are
     * stacked
     * @return the result `NDArray`. The stacked `NDArray` has one more dimension than
     * the input `NDArray`.
     */
    /**
     * Joins a `NDArray` along the first axis.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array1 = manager.create(new float[] {0f, 1f});
     * jshell&gt; NDArray array2 = manager.create(new float[] {2f, 3f});
     * jshell&gt; array1.stack(array2)
     * ND: (2, 2) cpu() float32
     * [[0., 1.],
     * [2., 3.],
     * ]
    </pre> *
     *
     * @param array the input `NDArray` which must have the same [Shape]as this `NDArray`
     * @return the result `NDArray`. The stacked `NDArray` has one more dimension than
     * the input `NDArray`.
     */
    //    @JvmOverloads
    fun stack(array: NDArray, axis: Int = 0): NDArray = nDArrayInternal.stack(NDList(array), axis)

    /**
     * Joins a `NDArray` along the first axis.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array1 = manager.create(new float[] {0f, 1f});
     * jshell&gt; NDArray array2 = manager.create(new float[] {2f, 3f});
     * jshell&gt; array1.concat(array2)
     * ND: (4) cpu() float32
     * [0., 1., 2., 3.]
    </pre> *
     *
     * @param array a `NDArray` which have the same [Shape]as this `NDArray`,
     * except in the dimension corresponding to axis
     * @return the concatenated `NDArray`
     */
    infix fun concat(array: NDArray): NDArray = concat(array, 0)

    /**
     * Joins a `NDArray` along an existing axis.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array1 = manager.create(new float[] {0f, 1f});
     * jshell&gt; NDArray array2 = manager.create(new float[] {2f, 3f});
     * jshell&gt; array1.concat(array2, 0);
     * ND: (4) cpu() float32
     * [0., 1., 2., 3.]
    </pre> *
     *
     * @param array a `NDArray` which have the same [Shape]as this `NDArray`,
     * except in the dimension corresponding to axis
     * @param axis the axis along which this `NDArray` will be joined
     * @return the concatenated `NDArray`
     */
    /**
     * Joins a `NDArray` along the first axis.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array1 = manager.create(new float[] {0f, 1f});
     * jshell&gt; NDArray array2 = manager.create(new float[] {2f, 3f});
     * jshell&gt; array1.concat(array2)
     * ND: (4) cpu() float32
     * [0., 1., 2., 3.]
    </pre> *
     *
     * @param array a `NDArray` which have the same [Shape]as this `NDArray`,
     * except in the dimension corresponding to axis
     * @return the concatenated `NDArray`
     */
    fun concat(array: NDArray, axis: Int = 0): NDArray = nDArrayInternal.concat(NDList(array), axis)

    ////////////////////////////////////////
    // Operations: Logical Op
    ////////////////////////////////////////
    /**
     * Returns the truth value of this `NDArray` AND the other `NDArray` element-wise.
     *
     *
     * The shapes of this `NDArray` and the other `NDArray` must be broadcastable.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array1 = manager.create(new boolean[] {true});
     * jshell&gt; NDArray array2 = manager.create(new boolean[] {false});
     * jshell&gt; array1.logicalAnd(array2);
     * ND: (1) cpu() boolean
     * [false]
     * jshell&gt; array1 = manager.create(new boolean[] {true, false});
     * jshell&gt; array2 = manager.create(new boolean[] {false, false});
     * jshell&gt; array1.logicalAnd(array2);
     * ND: (2) cpu() boolean
     * [false, false]
    </pre> *
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(5f);
     * jshell&gt; array.gt(1).logicalAnd(array.lt(4));
     * ND: (5) cpu() boolean
     * [false, false,  true,  true, false]
    </pre> *
     *
     * @param other the other `NDArray` to operate on
     * @return the boolean `NDArray` of the logical AND operation applied to the elements of
     * this `NDArray` and the other `NDArray`
     */
    infix fun logicalAnd(other: NDArray): NDArray

    /**
     * Computes the truth value of this `NDArray` OR the other `NDArray` element-wise.
     *
     *
     * The shapes of this `NDArray` and the other `NDArray` must be broadcastable.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array1 = manager.create(new boolean[] {true});
     * jshell&gt; NDArray array2 = manager.create(new boolean[] {false});
     * jshell&gt; array1.logicalOr(array2);
     * ND: (1) cpu() boolean
     * [ true]
     * jshell&gt; array1 = manager.create(new boolean[] {true, false});
     * jshell&gt; array2 = manager.create(new boolean[] {false, false});
     * jshell&gt; array1.logicalOr(array2);
     * ND: (2) cpu() boolean
     * [ true, false]
    </pre> *
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(5f);
     * jshell&gt; array.lt(1).logicalOr(array.gt(3));
     * ND: (5) cpu() boolean
     * [ true, false, false, false,  true]
    </pre> *
     *
     * @param other the other `NDArray` to operate on
     * @return the boolean `NDArray` of the logical OR operation applied to the elements of
     * this `NDArray` and the other `NDArray`
     */
    infix fun logicalOr(other: NDArray): NDArray

    /**
     * Computes the truth value of this `NDArray` XOR the other `NDArray` element-wise.
     *
     *
     * The shapes of this `NDArray` and the other `NDArray` must be broadcastable.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new boolean[] {true});
     * jshell&gt; array1.logicalXor(array2);
     * ND: (1) cpu() boolean
     * [ true]
     * jshell&gt; array1 = manager.create(new boolean[] {true, false});
     * jshell&gt; array2 = manager.create(new boolean[] {false, false});
     * jshell&gt; array1.logicalXor(array2);
     * ND: (2) cpu() boolean
     * [ true, false]
    </pre> *
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(5f);
     * jshell&gt; array.lt(1).logicalXor(array.gt(3));
     * ND: (5) cpu() boolean
     * [ true, false, false, false,  true]
    </pre> *
     *
     * @param other the other `NDArray` to operate on
     * @return the boolean `NDArray` of the logical XOR operation applied to the elements of
     * this `NDArray` and the other `NDArray`
     */
    infix fun logicalXor(other: NDArray): NDArray

    /**
     * Computes the truth value of NOT this `NDArray` element-wise.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new boolean[] {true});
     * jshell&gt; array.logicalNot();
     * ND: (1) cpu() boolean
     * [ false]
    </pre> *
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(5f);
     * jshell&gt; array.lt(1).logicalNot();
     * ND: (5) cpu() boolean
     * [false, true, true,  true,  true]
    </pre> *
     *
     * @return the boolean `NDArray`
     */
    operator fun not(): NDArray

    ////////////////////////////////////////
    // Operations: Other
    ////////////////////////////////////////
    /**
     * Returns the indices that would sort this `NDArray`.
     *
     *
     * Perform an indirect sort along the given axis. It returns a `NDArray` of indices of
     * the same [Shape] as this `NDArray`.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {3f, 1f, 2f});
     * jshell&gt; array.argSort();
     * ND: (3) cpu() int64
     * [ 1,  2,  0]
     *
     * jshell&gt; array = manager.create(new float[] {0f, 3f, 2f, 2f}, new Shape(2, 2));
     * jshell&gt; array.argSort();
     * ND: (2, 2) cpu() int64
     * [[ 0,  1],
     * [ 0,  1],
     * ]
    </pre> *
     *
     * @return a `NDArray` of indices corresponding to elements in this `NDArray` on the
     * axis, the output DataType is always [DataType.INT64]
     * @see NDArray.argSort
     */
    fun argSort(): NDArray = argSort(-1, true)

    /**
     * Returns the indices that would sort this `NDArray` given the axis.
     *
     *
     * Perform an indirect sort along the given axis. It returns a `NDArray` of indices of
     * the same [Shape] as this `NDArray`.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {0f, 3f, 2f, 2f}, new Shape(2, 2));
     * jshell&gt; array.argSort(0);
     * ND: (2, 2) cpu() int64
     * [[ 0,  1],
     * [ 1,  0],
     * ]
     * jshell&gt; array.argSort(1);
     * ND: (2, 2) cpu() int64
     * [[ 0,  1],
     * [ 0,  1],
     * ]
    </pre> *
     *
     * @param axis the axis to sort along
     * @return a `NDArray` of indices corresponding to elements in this `NDArray` on the
     * axis, the output DataType is always [DataType.INT64]
     * @see NDArray.argSort
     */
    infix fun argSort(axis: Int): NDArray = argSort(axis, true)

    /**
     * Returns the indices that would sort this `NDArray` given the axis.
     *
     *
     * Perform an indirect sort along the given axis. It returns a `NDArray` of indices of
     * the same [Shape] as this `NDArray`.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {0f, 3f, 2f, 2f}, new Shape(2, 2));
     * jshell&gt; array.argSort(0, false);
     * ND: (2, 2) cpu() int64
     * [[ 1,  0],
     * [ 0,  1],
     * ]
    </pre> *
     *
     * @param axis the axis to sort along
     * @param ascending whether to sort ascending
     * @return a `NDArray` of indices corresponding to elements in this `NDArray` on the
     * axis, the output DataType is always [DataType.INT64]
     */
    fun argSort(axis: Int, ascending: Boolean): NDArray

    /**
     * Sorts the flattened `NDArray`.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {1f, 4f, 3f, 1f}, new Shape(2, 2));
     * jshell&gt; array;
     * ND: (2, 2) cpu() float32
     * [[1., 4.],
     * [3., 1.],
     * ]
     * jshell&gt; array.sort(); // sort the flattened array
     * ND: (2, 2) cpu() float32
     * [[1., 4.],
     * [1., 3.],
     * ]
    </pre> *
     *
     * @return the sorted `NDArray`
     */
    fun sort(): NDArray

    /**
     * Sorts the flattened `NDArray`.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {1f, 4f, 3f, 1f}, new Shape(2, 2));
     * jshell&gt; array;
     * ND: (2, 2) cpu() float32
     * [[1., 4.],
     * [3., 1.],
     * ]
     * jshell&gt; array.sort(0); // sort along the first axis
     * ND: (2, 2) cpu() float32
     * [[1., 1.],
     * [3., 4.],
     * ]
    </pre> *
     *
     * @param axis the axis to sort along
     * @return the sorted `NDArray`
     */
    infix fun sort(axis: Int): NDArray

    /**
     * Applies the softmax function along the given axis.
     *
     * @param axis the axis along which to apply
     * @return the result `NDArray`
     * @see [softmax](https://en.wikipedia.org/wiki/Softmax_function)
     *
     * @see NDArray.softmax
     */
    infix fun softmax(axis: Int): NDArray

    /**
     * Applies the softmax function followed by a logarithm.
     *
     *
     * Mathematically equivalent to calling softmax and then log. This single operator is faster
     * than calling two operators and numerically more stable when computing gradients.
     *
     * @param axis the axis along which to apply
     * @return the result `NDArray`
     */
    infix fun logSoftmax(axis: Int): NDArray

    /**
     * Returns the cumulative sum of the elements in the flattened `NDArray`.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {1f, 2f, 3f, 4f, 5f, 6f}, new Shape(2, 3));
     * jshell&gt; array;
     * ND: (2, 3) cpu() float32
     * [[1., 2., 3.],
     * [4., 5., 6.],
     * ]
     * jshell&gt; array.cumSum(); // cumSum on flattened array
     * ND: (6) cpu() float32
     * [ 1.,  3.,  6., 10., 15., 21.]
    </pre> *
     *
     * @return the cumulative sum of the elements in the flattened `NDArray`
     */
    fun cumSum(): NDArray

    /**
     * Return the cumulative sum of the elements along a given axis.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {1f, 2f, 3f, 4f, 5f, 6f}, new Shape(2, 3));
     * jshell&gt; array;
     * ND: (2, 3) cpu() float32
     * [[1., 2., 3.],
     * [4., 5., 6.],
     * ]
     * jshell&gt; array.cumSum(0);
     * ND: (2, 3) cpu() float32
     * [[1., 2., 3.],
     * [5., 7., 9.],
     * ]
     * jshell&gt; array.cumSum(1);
     * ND: (2, 3) cpu() float32
     * [[ 1.,  3.,  6.],
     * [ 4.,  9., 15.],
     * ]
    </pre> *
     *
     * @param axis the axis along which the cumulative sum is computed
     * @return the cumulative sum along the specified axis
     */
    infix fun cumSum(axis: Int): NDArray

    /**
     * Replace the handle of the NDArray with the other. The NDArray used for replacement will be
     * killed.
     *
     *
     * Please use with caution, this method will make the input argument unusable.
     *
     * @param replaced the handle provider that will be killed
     */
    infix fun intern(replaced: NDArray)

    /**
     * Returns the boolean `NDArray` with value `true` where this `NDArray`'s
     * entries are infinite, or `false` where they are not infinite.
     *
     * @return the boolean `NDArray` with value `true` if this `NDArray`'s entries
     * are infinite
     */
    //    @JvmField
    val isInfinite: NDArray

    /**
     * Computes the inverse of square `NDArray` if it exists.
     *
     * @return the inverse of square `NDArray`.
     */
    fun inverse(): NDArray

    /**
     * Returns the boolean `NDArray` with value `true` where this `NDArray`'s
     * entries are NaN, or `false` where they are not NaN.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {Float.POSITIVE_INFINITY, 0, Float.NaN});
     * jshell&gt; array.isNaN();
     * ND: (3) cpu() boolean
     * [false, false,  true]
    </pre> *
     *
     * @return the boolean `NDArray` with value `true` if this `NDArray`'s [     ] are NaN
     */
    //    @JvmField
    val isNaN: NDArray

    /**
     * Constructs a `NDArray` by repeating this `NDArray` the number of times given
     * repeats.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {0f, 1f, 2f});
     * jshell&gt; array.tile(2);
     * ND: (6) cpu() float32
     * [0., 1., 2., 0., 1., 2.]
    </pre> *
     *
     * @param repeats the number of times to repeat for each dimension
     * @return a NDArray that has been tiled
     */
    fun tile(repeats: Long): NDArray

    /**
     * Constructs a `NDArray` by repeating this `NDArray` the number of times given by
     * repeats along given axis.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {0f, 1f, 2f});
     * jshell&gt; array.tile(1, 2);
     * ND: (1, 6) cpu() float32
     * [[0., 1., 2., 0., 1., 2.],
     * ]
    </pre> *
     *
     * @param axis the axis to repeat
     * @param repeats the number of times to repeat for each axis
     * @return a `NDArray` that has been tiled
     * @throws IllegalArgumentException thrown for invalid axis
     */
    fun tile(axis: Int, repeats: Long): NDArray

    /**
     * Constructs a `NDArray` by repeating this `NDArray` the number of times given by
     * repeats.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {0f, 1f, 2f});
     * jshell&gt; array.tile(new long[] {2, 2});
     * ND: (2, 6) cpu() float32
     * [[0., 1., 2., 0., 1., 2.],
     * [0., 1., 2., 0., 1., 2.],
     * ]
    </pre> *
     *
     * @param repeats the number of times to repeat along each axis
     * @return a `NDArray` that has been tiled
     */
    infix fun tile(repeats: LongArray): NDArray

    /**
     * Constructs a `NDArray` by repeating this `NDArray` the number of times to match
     * the desired shape.
     *
     *
     * If the desired [Shape]has fewer dimensions than this `NDArray`, it will tile
     * against the last axis.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {0f, 1f, 2f});
     * jshell&gt; array.tile(new Shape(6));
     * ND: (6) cpu() float32
     * [0., 1., 2., 0., 1., 2.]
    </pre> *
     *
     * @param desiredShape the [Shape]that should be converted to
     * @return a `NDArray` that has been tiled
     */
    infix fun tile(desiredShape: Shape): NDArray

    /**
     * Repeats element of this `NDArray` the number of times given repeats.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {0f, 1f, 2f});
     * jshell&gt; array.repeat(2);
     * ND: (6) cpu() float32
     * [0., 0., 1., 1., 2., 2.]
    </pre> *
     *
     * @param repeats the number of times to repeat for each axis
     * @return an `NDArray` that has been repeated
     */
    infix fun repeat(repeats: Long): NDArray

    /**
     * Repeats element of this `NDArray` the number of times given repeats along given axis.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {0f, 1f, 2f, 3f}, new Shape(2, 2));
     * jshell&gt; array.repeat(1, 2);
     * ND: (6) cpu() float32
     * [[0., 0., 1., 1.],
     * [2., 2., 3., 3.]]
    </pre> *
     *
     * @param axis the axis to repeat
     * @param repeats the number of times to repeat for each axis
     * @return an `NDArray` that has been repeated
     * @throws IllegalArgumentException thrown for invalid axis
     */
    fun repeat(axis: Int, repeats: Long): NDArray

    /**
     * Repeats element of this `NDArray` the number of times given repeats along each axis.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {0f, 1f, 2f, 3f}, new Shape(2, 2));
     * jshell&gt; array.repeat(new long[] {2, 2});
     * ND: (12) cpu() float32
     * [0., 0., 0., 0., 1., 1., 1., 1., 2., 2., 2., 2.]
    </pre> *
     *
     * @param repeats the number of times to repeat along each axis
     * @return a `NDArray` that has been repeated
     */
    fun repeat(repeats: LongArray): NDArray

    /**
     * Repeats element of this `NDArray` to match the desired shape.
     *
     *
     * If the desired [Shape] has fewer dimensions that the array, it will repeat against
     * the last axis.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {0f, 1f, 2f, 3f}, new Shape(2, 2));
     * jshell&gt; array.repeat(new Shape(4, 4));
     * ND: (4, 4) cpu() float32
     * [[0., 0., 1., 1.],
     * [0., 0., 1., 1.],
     * [2., 2., 3., 3.],
     * [2., 2., 3., 3.],
     * ]
    </pre> *
     *
     * @param desiredShape the [Shape] that should be converted to
     * @return an `NDArray` that has been repeated
     */
    fun repeat(desiredShape: Shape): NDArray

    /**
     * Dot product of this `NDArray` and the other `NDArray`.
     *
     *
     *  * If both this `NDArray` and the other `NDArray` are 1-D `NDArray`s, it
     * is inner product of vectors (without complex conjugation).
     *  * If both this `NDArray` and the other `NDArray` are 2-D `NDArray`s, it
     * is matrix multiplication.
     *  * If either this `NDArray` or the other `NDArray` is 0-D `NDArray`
     * (scalar), it is equivalent to mul.
     *  * If this `NDArray` is N-D `NDArray` and the other `NDArray` is 1-D
     * `NDArray`, it is a sum product over the last axis of those.
     *  * If this `NDArray` is N-D `NDArray` and the other `NDArray` is M-D
     * `NDArray`(where M&gt;&#61;2), it is a sum product over the last axis of this
     * `NDArray` and the second-to-last axis of the other `NDArray`
     *
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array1 = manager.create(new float[] {1f, 2f, 3f});
     * jshell&gt; NDArray array2 = manager.create(new float[] {4f, 5f, 6f});
     * jshell&gt; array1.dot(array2); // inner product
     * ND: () cpu() float32
     * 32.
     * jshell&gt; array1 = manager.create(new float[] {1f, 2f, 3f, 4f}, new Shape(2, 2));
     * jshell&gt; array2 = manager.create(new float[] {5f, 6f, 7f, 8f}, new Shape(2, 2));
     * jshell&gt; array1.dot(array2); // matrix multiplication
     * ND: (2, 2) cpu() float32
     * [[19., 22.],
     * [43., 50.],
     * ]
     * jshell&gt; array1 = manager.create(new float[] {1f, 2f, 3f, 4f}, new Shape(2, 2));
     * jshell&gt; array2 = manager.create(5f);
     * jshell&gt; array1.dot(array2);
     * ND: (2, 2) cpu() float32
     * [[ 5., 10.],
     * [15., 20.],
     * ]
     * jshell&gt; array1 = manager.create(new float[] {1f, 2f, 3f, 4f}, new Shape(2, 2));
     * jshell&gt; array2 = manager.create(new float[] {1f, 2f});
     * jshell&gt; array1.dot(array2);
     * ND: (2) cpu() float32
     * [ 5., 11.]
     * jshell&gt; array1 = manager.create(new float[] {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f}, new Shape(2, 2, 2));
     * jshell&gt; array2 = manager.create(new float[] {1f, 2f, 3f ,4f}, new Shape(2, 2));
     * jshell&gt; array1.dot(array2);
     * ND: (2, 2, 2) cpu() float32
     * [[[ 7., 10.],
     * [15., 22.],
     * ],
     * [[23., 34.],
     * [31., 46.],
     * ],
     * ]
    </pre> *
     *
     * @param other the other `NDArray` to perform dot product with
     * @return the result `NDArray`
     */
    infix fun `·`(other: NDArray): NDArray = dot(other)

    /**
     * Dot product of this `NDArray` and the other `NDArray`.
     *
     *
     *  * If both this `NDArray` and the other `NDArray` are 1-D `NDArray`s, it
     * is inner product of vectors (without complex conjugation).
     *  * If both this `NDArray` and the other `NDArray` are 2-D `NDArray`s, it
     * is matrix multiplication.
     *  * If either this `NDArray` or the other `NDArray` is 0-D `NDArray`
     * (scalar), it is equivalent to mul.
     *  * If this `NDArray` is N-D `NDArray` and the other `NDArray` is 1-D
     * `NDArray`, it is a sum product over the last axis of those.
     *  * If this `NDArray` is N-D `NDArray` and the other `NDArray` is M-D
     * `NDArray`(where M&gt;&#61;2), it is a sum product over the last axis of this
     * `NDArray` and the second-to-last axis of the other `NDArray`
     *
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array1 = manager.create(new float[] {1f, 2f, 3f});
     * jshell&gt; NDArray array2 = manager.create(new float[] {4f, 5f, 6f});
     * jshell&gt; array1.dot(array2); // inner product
     * ND: () cpu() float32
     * 32.
     * jshell&gt; array1 = manager.create(new float[] {1f, 2f, 3f, 4f}, new Shape(2, 2));
     * jshell&gt; array2 = manager.create(new float[] {5f, 6f, 7f, 8f}, new Shape(2, 2));
     * jshell&gt; array1.dot(array2); // matrix multiplication
     * ND: (2, 2) cpu() float32
     * [[19., 22.],
     * [43., 50.],
     * ]
     * jshell&gt; array1 = manager.create(new float[] {1f, 2f, 3f, 4f}, new Shape(2, 2));
     * jshell&gt; array2 = manager.create(5f);
     * jshell&gt; array1.dot(array2);
     * ND: (2, 2) cpu() float32
     * [[ 5., 10.],
     * [15., 20.],
     * ]
     * jshell&gt; array1 = manager.create(new float[] {1f, 2f, 3f, 4f}, new Shape(2, 2));
     * jshell&gt; array2 = manager.create(new float[] {1f, 2f});
     * jshell&gt; array1.dot(array2);
     * ND: (2) cpu() float32
     * [ 5., 11.]
     * jshell&gt; array1 = manager.create(new float[] {1f, 2f, 3f, 4f, 5f, 6f, 7f, 8f}, new Shape(2, 2, 2));
     * jshell&gt; array2 = manager.create(new float[] {1f, 2f, 3f ,4f}, new Shape(2, 2));
     * jshell&gt; array1.dot(array2);
     * ND: (2, 2, 2) cpu() float32
     * [[[ 7., 10.],
     * [15., 22.],
     * ],
     * [[23., 34.],
     * [31., 46.],
     * ],
     * ]
    </pre> *
     *
     * @param other the other `NDArray` to perform dot product with
     * @return the result `NDArray`
     */
    infix fun dot(other: NDArray): NDArray

    /**
     * Product matrix of this `NDArray` and the other `NDArray`.
     *
     *
     * The behavior depends on the arguments in the following way.
     *
     *
     *  * If both this `NDArray` and the other `NDArray` are 2-D `NDArray`s,
     * they are multiplied like conventional matrices
     *  * If either this `NDArray` or the other `NDArray` is N-D `NDArray`, N
     * &gt; 2 , it is treated as a stack of matrices residing in the last two indexes and
     * broadcast accordingly.
     *  * If this `NDArray` is 1-D `NDArray`, it is promoted to a matrix by
     * prepending a 1 to its dimensions. After matrix multiplication the prepended 1 is
     * removed.
     *  * If other `NDArray` is 1-D `NDArray`, it is promoted to a matrix by
     * appending a 1 to its dimensions. After matrix multiplication the appended 1 is removed.
     *
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array1 = manager.create(new float[] {1f, 0f, 0f, 1f}, new Shape(2, 2));
     * jshell&gt; NDArray array2 = manager.create(new float[] {4f, 1f, 2f, 2f}, new Shape(2, 2));
     * jshell&gt; array1.matMul(array2); // for 2-D arrays, it is the matrix product
     * ND: (2, 2) cpu() float32
     * [[4., 1.],
     * [2., 2.],
     * ]
     * jshell&gt; array1 = manager.create(new float[] {1f, 0f, 0f, 1f}, new Shape(2, 2));
     * jshell&gt; array2 = manager.create(new float[] {1f, 2f});
     * jshell&gt; array1.matMul(array2);
     * ND: (2) cpu() float32
     * [1., 2.]
     * jshell&gt; array1 = manager.create(new float[] {1f, 0f, 0f, 1f}, new Shape(2, 2));
     * jshell&gt; array2 = manager.create(new float[] {1f, 2f});
     * jshell&gt; array1.matMul(array2);
     * ND: (2) cpu() float32
     * [1., 2.]
     * jshell&gt; array1 = manager.arange(2f * 2f * 4f).reshape(2, 2, 4);
     * jshell&gt; array2 = manager.arange(2f * 2f * 4f).reshape(2, 4, 2);
     * jshell&gt; array1.matMul(array2).get("0, 1, 1");
     * ND: () cpu() float32
     * 98.
    </pre> *
     *
     * @param other the other `NDArray` to perform matrix product with
     * @return the result `NDArray`
     */
    infix fun matTimes(other: NDArray): NDArray

    /**
     * Batch product matrix of this `NDArray` and the other `NDArray`.
     *
     * @param other the other `NDArray` to perform matrix product with
     * @return the result `NDArray`
     */
    infix fun batchMatTimes(other: NDArray): NDArray

    /**
     * Clips (limit) the values in this `NDArray`.
     *
     *
     * Given an interval, values outside the interval are clipped to the interval edges. For
     * example, if an interval of [0, 1] is specified, values smaller than 0 become 0, and values
     * larger than 1 become 1.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(10f);
     * jshell&gt; array.clip(1, 8);
     * ND: (10) cpu() float32
     * [1., 1., 2., 3., 4., 5., 6., 7., 8., 8.]
    </pre> *
     *
     * @param min the minimum value
     * @param max the maximum value
     * @return an `NDArray` with the elements of this `NDArray`, but where values &lt;
     * min are replaced with min, and those &gt; max with max
     */
    fun clip(min: Number, max: Number): NDArray

    /**
     * Interchanges two axes of this `NDArray`.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {1f, 2f ,3f}, new Shape(1, 3));
     * jshell&gt; array;
     * ND: (1, 3) cpu() float32
     * [[1., 2., 3.],
     * ]
     * jshell&gt; array.swapAxes(0, 1);
     * ND: (3, 1) cpu() float32
     * [[1.],
     * [2.],
     * [3.],
     * ]
    </pre> *
     *
     * @param axis1 the first axis
     * @param axis2 the second axis
     * @return the swapped axes `NDArray`
     */
    fun swapAxes(axis1: Int, axis2: Int): NDArray {
        val dims = IntStream.range(0, shape.dimension()).toArray()
        val tmp = dims[axis1]
        dims[axis1] = dims[axis2]
        dims[axis2] = tmp
        return transpose(*dims)
    }

    /**
     * Returns the reverse order of elements in an array along the given axis.
     *
     *
     * The shape of the array is preserved, but the elements are reordered.
     *
     * @param axes the axes to flip on
     * @return the newly flipped array
     */
    fun flip(vararg axes: Int): NDArray

    /**
     * Returns this `NDArray` with axes transposed.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {1f, 2f ,3f, 4f}, new Shape(2, 2));
     * jshell&gt; array;
     * ND: (2, 2) cpu() float32
     * [[1., 2.],
     * [3., 4.],
     * ]
     * jshell&gt; array.transpose();
     * ND: (2, 2) cpu() float32
     * [[1., 3.],
     * [2., 4.],
     * ]
    </pre> *
     *
     * @return the newly permuted array
     */
    fun transpose(): NDArray

    /**
     * Returns this `NDArray` with given axes transposed.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {1f, 2f ,3f, 4f}, new Shape(2, 2));
     * jshell&gt; array;
     * ND: (2, 2) cpu() float32
     * [[1., 2.],
     * [3., 4.],
     * ]
     * jshell&gt; array.transpose(1, 0);
     * ND: (2, 2) cpu() float32
     * [[1., 3.],
     * [2., 4.],
     * ]
     * jshell&gt; array = manager.arange(8f).reshape(2, 2, 2);
     * jshell&gt; array;
     * ND: (2, 2, 2) cpu() float32
     * [[[0., 1.],
     * [2., 3.],
     * ],
     * [[4., 5.],
     * [6., 7.],
     * ],
     * ]
     * jshell&gt; array.transpose(1, 0, 2);
     * ND: (2, 2, 2) cpu() float32
     * [[[0., 1.],
     * [4., 5.],
     * ],
     * [[2., 3.],
     * [6., 7.],
     * ],
     * ]
    </pre> *
     *
     * @param axes the axes to swap to
     * @return the transposed `NDArray`
     * @throws IllegalArgumentException thrown when passing a axis that is greater than the actual
     * number of dimensions
     */
    fun transpose(vararg axes: Int): NDArray

    /**
     * Broadcasts this `NDArray` to be the given shape.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {1f, 2f ,3f, 4f}, new Shape(2, 2));
     * jshell&gt; array;
     * ND: (2, 2) cpu() float32
     * [[1., 2.],
     * [3., 4.],
     * ]
     * jshell&gt; array.broadcast(new Shape(2, 2, 2));
     * ND: (2, 2, 2) cpu() float32
     * [[[1., 2.],
     * [3., 4.],
     * ],
     * [[1., 2.],
     * [3., 4.],
     * ],
     * ]
    </pre> *
     *
     * @param shape the new [Shape] of this `NDArray`
     * @return the broadcasted `NDArray`
     */
    infix fun broadcast(shape: Shape): NDArray

    /**
     * Broadcasts this `NDArray` to be the given shape.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {1f, 2f ,3f, 4f}, new Shape(2, 2));
     * jshell&gt; array;
     * ND: (2, 2) cpu() float32
     * [[1., 2.],
     * [3., 4.],
     * ]
     * jshell&gt; array.broadcast(2, 2, 2);
     * ND: (2, 2, 2) cpu() float32
     * [[[1., 2.],
     * [3., 4.],
     * ],
     * [[1., 2.],
     * [3., 4.],
     * ],
     * ]
    </pre> *
     *
     * @param shape the new [Shape] of this `NDArray`
     * @return the broadcasted `NDArray`
     */
    fun broadcast(vararg shape: Long): NDArray = broadcast(Shape(*shape))

    /**
     * Returns the indices of the maximum values into the flattened `NDArray`.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(6f).reshape(2, 3);
     * jshell&gt; array;
     * ND: (2, 3) cpu() float32
     * [[0., 1., 2.],
     * [3., 4., 5.],
     * ]
     * jshell&gt; array.argMax();
     * ND: () cpu() int64
     * 5.
    </pre> *
     *
     * @return a `NDArray` containing indices
     */
    fun argMax(): NDArray

    /**
     * Returns the indices of the maximum values along given axis.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(6f).reshape(2, 3);
     * jshell&gt; array;
     * ND: (2, 3) cpu() float32
     * [[0., 1., 2.],
     * [3., 4., 5.],
     * ]
     * jshell&gt; array.argMax(0);
     * ND: (3) cpu() int64
     * [1, 1, 1]
     * jshell&gt; array.argMax(1);
     * ND: (2) cpu() int64
     * [2, 2]
    </pre> *
     *
     * @param axis the axis along which to find maximum values
     * @return a `NDArray` containing indices
     */
    infix fun argMax(axis: Int): NDArray

    /**
     * Returns (values, indices) of the top k values along given axis.
     *
     * @param k the number of returned values
     * @param axis the axis to sort along, whose shape is reduced to k
     * @return a `NDList` containing (values, indices)
     */
    fun topK(k: Int, axis: Int): NDList = topK(k, axis, largest = true, sorted = true)

    /**
     * Returns (values, indices) of the top k values along given axis.
     *
     * @param k the number of returned values
     * @param axis the axis to sort along, whose shape is reduced to k
     * @param largest whether the largest or the smallest
     * @param sorted whether the sorted or not
     * @return a `NDList` containing (values, indices)
     */
    fun topK(k: Int, axis: Int, largest: Boolean, sorted: Boolean): NDList

    /**
     * Returns the indices of the minimum values into the flattened `NDArray`.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(6f).reshape(2, 3);
     * jshell&gt; array;
     * ND: (2, 3) cpu() float32
     * [[0., 1., 2.],
     * [3., 4., 5.],
     * ]
     * jshell&gt; array.argMin();
     * ND: () cpu() int64
     * 0.
    </pre> *
     *
     * @return a `NDArray` containing indices
     */
    fun argMin(): NDArray

    /**
     * Returns the indices of the minimum values along given axis.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.arange(6f).reshape(2, 3);
     * jshell&gt; array;
     * ND: (2, 3) cpu() float32
     * [[0., 1., 2.],
     * [3., 4., 5.],
     * ]
     * jshell&gt; array.argMin(0);
     * ND: (3) cpu() int64
     * [0, 0, 0]
     * jshell&gt; array.argMin(1);
     * ND: (2) cpu() int64
     * [0, 0]
    </pre> *
     *
     * @param axis the axis along which to find minimum values
     * @return a `NDArray` containing indices
     */
    infix fun argMin(axis: Int): NDArray

    /**
     * Returns percentile for this `NDArray`.
     *
     * @param percentile the target percentile in range of 0..100
     * @return the result `NDArray`
     */
    infix fun percentile(percentile: Number): NDArray

    /**
     * Returns median along given dimension(s).
     *
     * @param percentile the target percentile in range of 0..100
     * @param axes the dimension to calculate percentile for
     * @return the result `NDArray` NDArray
     */
    fun percentile(percentile: Number, axes: IntArray): NDArray

    /**
     * Returns median value for this `NDArray`.
     *
     * @return the median `NDArray`
     */
    fun median(): NDArray

    /**
     * Returns median value along given axes.
     *
     * @param axes the axes along which to perform the median operation
     * @return the median `NDArray` along the specified axes
     */
    infix fun median(axes: IntArray): NDArray

    // ------------ Sparse methods ------------
    /**
     * Returns a dense representation of the sparse `NDArray`.
     *
     * @return the result `NDArray`
     */
    fun toDense(): NDArray

    /**
     * Returns a sparse representation of `NDArray`.
     *
     * @param fmt the [SparseFormat] of this `NDArray`
     * @return the result `NDArray`
     */
    infix fun toSparse(fmt: SparseFormat): NDArray

    /**
     * Returns the indices of elements that are non-zero.
     *
     *
     * Note that the behavior is slightly different from numpy.nonzero. Numpy returns a tuple of
     * NDArray, one for each dimension of NDArray. DJL nonzero returns only one `NDArray` with
     * last dimension containing all dimension of indices.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {1f, 1f, 1f, 0f, 1f});
     * jshell&gt; array.nonzero();
     * ND: (4, 1) cpu() int64
     * [[ 0],
     * [ 1],
     * [ 2],
     * [ 4],
     * ]
     * jshell&gt; array = manager.create(new float[] {3f, 0f, 0f, 0f, 4f, 0f, 5f, 6f, 0f}).reshape(3, 3);
     * jshell&gt; array;
     * ND: (3, 3) cpu() float32
     * [[3., 0., 0.],
     * [0., 4., 0.],
     * [5., 6., 0.],
     * ]
     * jshell&gt; array.nonzero();
     * ND: (4, 2) cpu() int64
     * [[ 0,  0],
     * [ 1,  1],
     * [ 2,  0],
     * [ 2,  1],
     * ]
    </pre> *
     *
     * @return the indices of the elements that are non-zero
     */
    fun nonzero(): NDArray

    val isEmpty: Boolean
        /**
         * Returns `true` if this `NDArray` is special case: no-value `NDArray`.
         *
         *
         * Examples
         *
         * <pre>
         * jshell&gt; NDArray array = manager.create(new Shape(2, 0, 1));
         * jshell&gt; array;
         * ND: (2, 0, 1) cpu() float32
         * []
         * jshell&gt; array.isEmpty();
         * true
        </pre> *
         *
         * @return `true` if this NDArray is empty
         */
        get() = shape.size() == 0L

    /**
     * Returns `true` if all elements within this `NDArray` are non-zero or `true`.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new boolean[] {true, false, true, true}, new Shape(2, 2));
     * jshell&gt; array.all();
     * ND: () cpu() boolean
     * false
     * jshell&gt; NDArray array = manager.create(new float[] {-1f, 4f, 5f});
     * jshell&gt; array.all(); // all elements are non-zero
     * ND: () cpu() boolean
     * true
    </pre> *
     *
     * @return `true` if all elements within this `NDArray` are non-zero or `true`
     */
    fun all(): NDArray {
        // result of sum operation is int64 now
        return toType(DataType.BOOLEAN, false).sum().eq(size)
    }

    /**
     * Returns `true` if any of the elements within this `NDArray` are non-zero or
     * `true`.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new boolean[] {true, false, true, true}, new Shape(2, 2));
     * jshell&gt; array.any();
     * ND: () cpu() boolean
     * true
     * jshell&gt; NDArray array = manager.create(new float[] {-1, 0, 5});
     * jshell&gt; array.any() // all elements are non-zero
     * ND: () cpu() boolean
     * true
    </pre> *
     *
     * @return `true` if any of the elements within this `NDArray` are non-zero or
     * `true`
     */
    fun any(): NDArray = toType(DataType.BOOLEAN, false).sum().gt(0)

    /**
     * Returns `true` if none of the elements within this `NDArray` are non-zero or
     * `true`.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new boolean[] {false, false});
     * jshell&gt; array.none();
     * ND: () cpu() boolean
     * true
     * jshell&gt; NDArray array = manager.create(new float[] {-1f, 0f, 5f});
     * jshell&gt; array.none() // all elements are non-zero
     * ND: () cpu() boolean
     * false
    </pre> *
     *
     * @return `true` if none of the elements within this `NDArray` are non-zero or
     * `true`
     */
    fun none(): NDArray = toType(DataType.BOOLEAN, false).sum().eq(0)

    /**
     * Counts the number of non-zero values in this `NDArray`.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {0f, 0f, 1f, 2f, 7f, 0f}, new Shape(2, 3));
     * jshell&gt; array.countNonzero()
     * ND: () cpu() int64
     * 3
    </pre> *
     *
     * @return the number of non-zero values in this `NDArray`
     */
    fun countNonzero(): NDArray = toType(DataType.BOOLEAN, false).sum()

    /**
     * Counts the number of non-zero values in this `NDArray` along a given axis.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {0f, 0f, 1f, 2f, 7f, 0f}, new Shape(2, 3));
     * jshell&gt; array;
     * ND: (2, 3) cpu() float32
     * [[0., 0., 1.],
     * [2., 7., 0.],
     * ]
     * jshell&gt; array.countNonzero(0);
     * ND: (3) cpu() int64
     * [ 1,  1,  1]
     * jshell&gt; array.countNonzero(1);
     * ND: (2) cpu() int64
     * [ 1,  2]
    </pre> *
     *
     * @param axis the axis to operate on
     * @return the number of non-zero values in this `NDArray` along a given axis
     */
    infix fun countNonzero(axis: Int): NDArray = toType(DataType.BOOLEAN, false).sum(intArrayOf(axis))

    /**
     * Returns element-wise inverse gauss error function of the `NDArray`.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {0f, 0.5f, -1f});
     * jshell&gt; array.erfinv();
     * ND: (3) cpu() float32
     * [0., 0.4769, -inf]
    </pre> *
     *
     * @return The inverse of gauss error of the `NDArray`, element-wise
     */
    fun erfinv(): NDArray

    /**
     * Returns element-wise gauss error function of the `NDArray`.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {0f, 0.4769f, Float.NEGATIVE_INFINITY});
     * jshell&gt; array.erf();
     * ND: (3) cpu() float32
     * [0., 0.5, -1]
    </pre> *
     *
     * @return The gauss error of the `NDArray`, element-wise
     */
    fun erf(): NDArray

    /** {@inheritDoc}  */
    override fun getResourceNDArrays(): List<NDArray> = listOf(this)

    /**
     * Returns an internal representative of Native `NDArray`.
     *
     *
     * This method should only be used by Engine provider
     *
     * @return an internal representative of Native `NDArray`
     */
    //    @JvmField
    val nDArrayInternal: NDArrayEx

    /**
     * Returns `true` if this NDArray has been released.
     *
     * @return `true` if this NDArray has been released
     */
    //    @JvmField
    val isReleased: Boolean

    /**
     * Runs the debug string representation of this `NDArray`.
     *
     * @return the debug string representation of this `NDArray`
     */
    fun toDebugString(): String = when {
        isReleased -> "This array is already closed"
        dataType == DataType.STRING -> toStringArray(StandardCharsets.UTF_8).contentToString()
        else -> NDFormat.format(this, 100, 10, 10, 20)
    }

    /**
     * Runs the debug string representation of this `NDArray`.
     *
     * @param withContent true to show the content of NDArray
     * @return the debug string representation of this `NDArray`
     */
    fun toDebugString(withContent: Boolean): String = toDebugString(1000, 10, 10, 20, withContent)

    /**
     * Runs the debug string representation of this `NDArray`.
     *
     * @param maxSize the maximum elements to print out
     * @param maxDepth the maximum depth to print out
     * @param maxRows the maximum rows to print out
     * @param maxColumns the maximum columns to print out
     * @param withContent true to show the content of NDArray
     * @return the debug string representation of this `NDArray`
     */
    fun toDebugString(maxSize: Int, maxDepth: Int, maxRows: Int, maxColumns: Int, withContent: Boolean): String = when {
        isReleased -> "This array is already closed"
        dataType == DataType.STRING -> toStringArray(StandardCharsets.UTF_8).contentToString()
        else -> NDFormat.format(this, maxSize, maxDepth, maxRows, maxColumns, withContent)
    }

    /** {@inheritDoc}  */
    override fun close()

    /**
     * Returns the norm of this `NDArray`.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {-3f, -4f});
     * jshell&gt; array.norm();
     * ND: () cpu() float32
     * 5.
     * jshell&gt; NDArray array = manager.create(new float[] {1f, 2f, 3f, 4f}, new Shape(2, 2));
     * jshell&gt; array.norm();
     * ND: () cpu() float32
     * 5.4472
    </pre> *
     *
     * @return the norm of this `NDArray`
     */
    fun norm(): NDArray = norm(false)

    /**
     * Returns the norm of this `NDArray`.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {-3f, -4f});
     * jshell&gt; array.norm(true);
     * ND: () cpu() float32
     * 5.
     * jshell&gt; NDArray array = manager.create(new float[] {1f, 2f, 3f, 4f}, new Shape(2, 2));
     * jshell&gt; array.norm(true);
     * ND: () cpu() float32
     * [[5.4772],
     * ]
    </pre> *
     *
     * @param keepDims If this is set to True, the axes which are normed over are left in the result
     * as dimensions with size one. With this option the result will broadcast correctly against
     * the original x.
     * @return the norm of this `NDArray`
     */
    infix fun norm(keepDims: Boolean): NDArray

    /**
     * Returns the norm of this `NDArray`.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {1f, 2f, 3f, 4f}, new Shape(2, 2));
     * jshell&gt; array.norm(new int[] {0}, true);
     * ND: (1, 2) cpu() float32
     * [[3.1623, 4.4721],
     * ]
     * jshell&gt; NDArray array = manager.create(new float[] {1f, 2f, 3f, 4f}, new Shape(2, 2));
     * jshell&gt; array.norm(new int[] {0}, false);
     * ND: (2) cpu() float32
     * [3.1623, 4.4721]
    </pre> *
     *
     * @param axes If axes contains an integer, it specifies the axis of x along which to compute
     * the vector norms. If axis contains 2 integers, it specifies the axes that hold 2-D
     * matrices, and the matrix norms of these matrices are computed.
     * @param keepDims keepDims If this is set to True, the axes which are normed over are left in
     * the result as dimensions with size one. With this option the result will broadcast
     * correctly against the original x.
     * @return the norm of this `NDArray`
     */
    /**
     * Returns the norm of this `NDArray`.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {-3f, -4f});
     * jshell&gt; array.norm(new int[] {0});
     * ND: () cpu() float32
     * 5.
     * jshell&gt; NDArray array = manager.create(new float[] {1f, 2f, 3f, 4f}, new Shape(2, 2));
     * jshell&gt; array.norm(new int[] {0});
     * ND: (2) cpu() float32
     * [3.1623, 4.4721]
    </pre> *
     *
     * @param axes If axes contains an integer, it specifies the axis of x along which to compute
     * the vector norms. If axis contains 2 integers, it specifies the axes that hold 2-D
     * matrices, and the matrix norms of these matrices are computed.
     * @return the norm of this `NDArray`
     */
    //    @JvmOverloads
    fun norm(axes: IntArray, keepDims: Boolean = false): NDArray = norm(2, axes, keepDims)

    /**
     * Returns the norm of this `NDArray`.
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new float[] {1f, 2f, 3f, 4f}, new Shape(2, 2));
     * jshell&gt; array.norm(2, new int[] {0}, true);
     * ND: (1, 2) cpu() float32
     * [[3.1623, 4.4721],
     * ]
     * jshell&gt; NDArray array = manager.create(new float[] {1f, 2f, 3f, 4f}, new Shape(2, 2));
     * jshell&gt; array.norm(2, new int[] {0}, false);
     * ND: (2) cpu() float32
     * [3.1623, 4.4721]
    </pre> *
     *
     * @param ord Order of the norm.
     * @param axes If axes contains an integer, it specifies the axis of x along which to compute
     * the vector norms. If axis contains 2 integers, it specifies the axes that hold 2-D
     * matrices, and the matrix norms of these matrices are computed.
     * @param keepDims keepDims If this is set to True, the axes which are normed over are left in
     * the result as dimensions with size one. With this option the result will broadcast
     * correctly against the original x.
     * @return the norm of this `NDArray`
     */
    fun norm(ord: Int, axes: IntArray, keepDims: Boolean): NDArray

    /**
     * Returns a one-hot `NDArray`.
     *
     *
     *  * The locations represented by indices take value 1, while all other locations take value
     * 0.
     *  * If the input `NDArray` is rank N, the output will have rank N+1. The new axis is
     * appended at the end.
     *  * If `NDArray` is a scalar the output shape will be a vector of length depth.
     *  * If `NDArray` is a vector of length features, the output shape will be features x
     * depth.
     *  * If `NDArray` is a matrix with shape [batch, features], the output shape will be
     * batch x features x depth.
     *
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new int[] {1, 0, 2, 0});
     * jshell&gt; array.oneHot(3);
     * ND: (4, 3) cpu() float32
     * [[0., 1., 0.],
     * [1., 0., 0.],
     * [0., 0., 1.],
     * [1., 0., 0.],
     * ]
     * jshell&gt; NDArray array = manager.create(new int[][] {{1, 0}, {1, 0}, {2, 0}});
     * jshell&gt; array.oneHot(3);
     * ND: (3, 2, 3) cpu() float32
     * [[[0., 1., 0.],
     * [1., 0., 0.],
     * ],
     * [[0., 1., 0.],
     * [1., 0., 0.],
     * ],
     * [[0., 0., 1.],
     * [1., 0., 0.],
     * ],
     * ]
    </pre> *
     *
     * @param depth Depth of the one hot dimension.
     * @return one-hot encoding of this `NDArray`
     * @see [Classification-problems](https://d2l.djl.ai/chapter_linear-networks/softmax-regression.html.classification-problems)
     */
    infix fun oneHot(depth: Int): NDArray = oneHot(depth, 1f, 0f, DataType.FLOAT32)

    /**
     * Returns a one-hot `NDArray`.
     *
     *
     *  * The locations represented by indices take value 1, while all other locations take value
     * 0.
     *  * If the input `NDArray` is rank N, the output will have rank N+1. The new axis is
     * appended at the end.
     *  * If `NDArray` is a scalar the output shape will be a vector of length depth.
     *  * If `NDArray` is a vector of length features, the output shape will be features x
     * depth.
     *  * If `NDArray` is a matrix with shape [batch, features], the output shape will be
     * batch x features x depth.
     *
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new int[] {1, 0, 2, 0});
     * jshell&gt; array.oneHot(3);
     * ND: (4, 3) cpu() float32
     * [[0., 1., 0.],
     * [1., 0., 0.],
     * [0., 0., 1.],
     * [1., 0., 0.],
     * ]
     * jshell&gt; NDArray array = manager.create(new int[][] {{1, 0}, {1, 0}, {2, 0}});
     * jshell&gt; array.oneHot(3);
     * ND: (3, 2, 3) cpu() float32
     * [[[0., 1., 0.],
     * [1., 0., 0.],
     * ],
     * [[0., 1., 0.],
     * [1., 0., 0.],
     * ],
     * [[0., 0., 1.],
     * [1., 0., 0.],
     * ],
     * ]
    </pre> *
     *
     * @param depth Depth of the one hot dimension.
     * @param dataType dataType of the output.
     * @return one-hot encoding of this `NDArray`
     * @see [Classification-problems](https://d2l.djl.ai/chapter_linear-networks/softmax-regression.html.classification-problems)
     */
    fun oneHot(depth: Int, dataType: DataType): NDArray = oneHot(depth, 1f, 0f, dataType)

    /**
     * Returns a one-hot `NDArray`.
     *
     *
     *  * The locations represented by indices take value onValue, while all other locations take
     * value offValue.
     *  * If the input `NDArray` is rank N, the output will have rank N+1. The new axis is
     * appended at the end.
     *  * If `NDArray` is a scalar the output shape will be a vector of length depth.
     *  * If `NDArray` is a vector of length features, the output shape will be features x
     * depth.
     *  * If `NDArray` is a matrix with shape [batch, features], the output shape will be
     * batch x features x depth.
     *
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array = manager.create(new int[] {1, 0, 2, 0});
     * jshell&gt; array.oneHot(3, 8f, 1f, array.getDataType());
     * ND: (4, 3) cpu() int32
     * [[ 1,  8,  1],
     * [ 8,  1,  1],
     * [ 1,  1,  8],
     * [ 8,  1,  1],
     * ]
    </pre> *
     *
     * @param depth Depth of the one hot dimension.
     * @param onValue The value assigned to the locations represented by indices.
     * @param offValue The value assigned to the locations not represented by indices.
     * @param dataType dataType of the output.
     * @return one-hot encoding of this `NDArray`
     * @see [Classification-problems](https://d2l.djl.ai/chapter_linear-networks/softmax-regression.html.classification-problems)
     */
    fun oneHot(depth: Int, onValue: Float, offValue: Float, dataType: DataType): NDArray

    /**
     * Batchwise product of this `NDArray` and the other `NDArray`.
     *
     *
     *  * batchDot is used to compute dot product of x and y when x and y are data in batch,
     * namely N-D (N greater or equal to 3) arrays in shape of (B0, …, B_i, :, :). For
     * example, given x with shape (B_0, …, B_i, N, M) and y with shape (B_0, …, B_i, M, K),
     * the result array will have shape (B_0, …, B_i, N, K), which is computed by:
     * batch_dot(x,y)[b_0, ..., b_i, :, :] = dot(x[b_0, ..., b_i, :, :], y[b_0, ..., b_i, :,
     * :])
     *
     *
     *
     * Examples
     *
     * <pre>
     * jshell&gt; NDArray array1 = manager.ones(new Shape(2, 1, 4));
     * jshell&gt; NDArray array2 = manager.ones(new Shape(2, 4, 6));
     * jshell&gt; array1.batchDot(array2);
     * ND: (2, 1, 6) cpu() float32
     * [[[4., 4., 4., 4., 4., 4.],
     * ],
     * [[4., 4., 4., 4., 4., 4.],
     * ],
     * ]
    </pre> *
     *
     * @param other the other `NDArray` to perform batch dot product with
     * @return the result `NDArray`
     */
    infix fun batchDot(other: NDArray): NDArray

    /**
     * Convert a general NDArray to its complex math format.
     *
     *
     * example: [10f, 12f] float32 -&gt; [10+12j] in complex64
     *
     * @return the complex NDArray
     */
    fun complex(): NDArray

    /**
     * Convert a complex NDArray to its real math format. example: [10+12j] in complex64 -&gt; [10f,
     * 12f] float32
     *
     * @return tje real NDArray
     */
    fun real(): NDArray

    companion object {
        /**
         * Decodes `NDArray` from bytes.
         *
         * @param manager [NDManager] used to create this `NDArray`
         * @param byteArray data used to decode
         * @return decoded `NDArray`
         */
        @JvmStatic
        fun decode(manager: NDManager, byteArray: ByteArray): NDArray = manager.decode(byteArray)
    }
}
