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
package ai.djl.ndarray

import ai.djl.Device
import ai.djl.engine.Engine
import ai.djl.ndarray.types.DataType
import ai.djl.ndarray.types.Shape
import ai.djl.util.Float16Utils
import ai.djl.util.PairList
import ai.djl.util.passthrough.PassthroughNDManager
import java.io.IOException
import java.io.InputStream
import java.nio.*
import java.nio.charset.Charset
import java.nio.charset.StandardCharsets
import java.nio.file.Path
import kotlin.math.cos

/**
 * NDArray managers are used to create <I>NDArrays</I> (n-dimensional array on native engine).
 *
 *
 * NDManager is implemented in each deep learning [Engine]. [NDArray]s are resources
 * that are allocated in each deep learning engine's native memory space. NDManager is the key class
 * that manages these native resources.
 *
 *
 * NDArray can only be created through NDManager. By default, NDArray's lifecycle is attached to
 * the creator NDManager. NDManager itself implements [AutoCloseable]. When NDManager is
 * closed, all the resource associated with it will be closed as well.
 *
 *
 * A typical place to obtain NDManager is in [Translator.processInput] or [Translator.processOutput].
 *
 *
 * The following is an example of how to use NDManager:
 *
 * <pre>
 * public class MyTranslator implements Translator&lt;FloatBuffer, String&gt; {
 *
 * &#064;Override
 * public NDList processInput(TranslatorContext ctx, FloatBuffer input) {
 * **NDManager manager = ctx.getNDManager();**
 * NDArray array = **manager**.create(shape);
 * array.set(input);
 * return new NDList(array);
 * } // NDArrays created in this method will be closed after method return.
 * }
</pre> *
 *
 *
 * NDManager has a hierarchical structure; it has a single parent NDManager and has child
 * NDManagers. When the parent NDManager is closed, all children will be closed as well.
 *
 *
 * The DJL engine manages NDManager's lifecycle by default. You only need to manage the user
 * created child NDManager. The child NDManager becomes useful when you create a large number of
 * temporary NDArrays and want to free the resources earlier than the parent NDManager's lifecycle.
 *
 *
 * The following is an example of such a use case:
 *
 * <pre>
 * public class MyTranslator implements Translator&lt;List&lt;FloatBuffer&gt;&gt;, String&gt; {
 *
 * &#064;Override
 * public NDList processInput(TranslatorContext ctx, List&lt;FloatBuffer&gt; input) {
 * NDManager manager = ctx.getNDManager();
 * NDArray array = manager.create(shape, dataType);
 * for (int i = 0; i &lt; input.size(); ++i) {
 * try (**NDManager childManager = manager.newSubManager()**) {
 * NDArray tmp = **childManager**.create(itemShape);
 * tmp.put(input.get(i);
 * array.put(i, tmp);
 * } // NDArray *tmp* will be closed here
 * }
 * return new NDList(array);
 * }
 * }
</pre> *
 *
 *
 * You can also close an individual NDArray. NDManager won't close an NDArray that's already been
 * closed. In certain use cases, you might want to return an NDArray outside of NDManager's scope.
 *
 * @see NDArray
 *
 * @see Translator
 *
 * @see TranslatorContext.getNDManager
 * @see [NDArray
 * Memory Management Guide](https://github.com/deepjavalibrary/djl/blob/master/docs/development/memory_management.md)
 */
interface NDManager : AutoCloseable {
    /**
     * Returns the default context used in Engine.
     *
     *
     * The default type is defined by whether the deep learning engine is recognizing GPUs
     * available on your machine. If there is no GPU available, CPU will be used.
     *
     * @return a [Device]
     */
    fun defaultDevice(): Device?

    /**
     * Allocates a new engine specific direct byte buffer.
     *
     * @param capacity the new buffer's capacity, in bytes
     * @return the new byte buffer
     */
    fun allocateDirect(capacity: Int): ByteBuffer

    /**
     * Creates a new `NDArray` if the input [NDArray] is from an external engine.
     *
     * @param array the input `NDArray`
     * @return a new `NDArray` if the input `NDArray` is from external engine
     */
    fun from(array: NDArray?): NDArray

    /**
     * Creates and initializes a scalar [NDArray].
     *
     * @param data the [Number] that needs to be set
     * @return a new instance of [NDArray]
     */
    fun create(data: Number): NDArray = when (data) {
        is Int -> create(data.toInt())
        is Float -> create(data.toFloat())
        is Double -> create(data.toDouble())
        is Long -> create(data.toLong())
        is Byte -> create(data.toByte())
        else -> throw IllegalArgumentException("Short conversion not supported!")
    }

    /**
     * Creates and initializes a scalar [NDArray].
     *
     * @param data the float that needs to be set
     * @return a new instance of [NDArray]
     */
    fun create(data: Float): NDArray = create(floatArrayOf(data), Shape())

    /**
     * Creates and initializes a scalar [NDArray].
     *
     * @param data the float data that needs to be set
     * @return a new instance of [NDArray]
     */
    fun create(data: Int): NDArray = create(intArrayOf(data), Shape())

    /**
     * Creates and initializes a scalar [NDArray].
     *
     * @param data the double data that needs to be set
     * @return a new instance of [NDArray]
     */
    fun create(data: Double): NDArray = create(doubleArrayOf(data), Shape())

    /**
     * Creates and initializes a scalar [NDArray].
     *
     * @param data the long data that needs to be set
     * @return a new instance of [NDArray]
     */
    fun create(data: Long): NDArray = create(longArrayOf(data), Shape())

    /**
     * Creates and initializes a scalar [NDArray].
     *
     * @param data the byte data that needs to be set
     * @return a new instance of [NDArray]
     */
    fun create(data: Byte): NDArray = create(byteArrayOf(data), Shape())

    /**
     * Creates and initializes a scalar [NDArray].
     *
     * @param data the boolean data that needs to be set
     * @return a new instance of [NDArray]
     */
    fun create(data: Boolean): NDArray = create(booleanArrayOf(data), Shape())

    /**
     * Creates and initializes a scalar [NDArray].
     *
     * @param data the String data that needs to be set
     * @return a new instance of [NDArray]
     */
    fun create(data: String): NDArray = create(arrayOf(data), StandardCharsets.UTF_8, Shape())

    /**
     * Creates and initializes 1D [NDArray].
     *
     * @param data the String data that needs to be set
     * @return a new instance of [NDArray]
     */
    fun create(data: Array<String>): NDArray = create(data, StandardCharsets.UTF_8)

    /**
     * Creates and initializes 1D [NDArray].
     *
     * @param data the String data that needs to be set
     * @param charset the charset to decode the string
     * @return a new instance of [NDArray]
     */
    fun create(data: Array<String>, charset: Charset): NDArray = create(data, charset, Shape(data.size.toLong()))

    /**
     * Creates a String [NDArray] based on the provided shape.
     *
     * @param data the flattened String array
     * @param shape the shape of the String NDArray
     * @return a new instance of `NDArray`
     */
    fun create(data: Array<String>, shape: Shape): NDArray = create(data, StandardCharsets.UTF_8, shape)

    /**
     * Creates a String [NDArray] based on the provided shape.
     *
     * @param data the flattened String array
     * @param charset the charset to decode the string
     * @param shape the shape of the String NDArray
     * @return a new instance of `NDArray`
     */
    fun create(data: Array<String>, charset: Charset, shape: Shape): NDArray

    /**
     * Creates and initializes a 1D [NDArray].
     *
     * @param data the bool array that needs to be set
     * @return a new instance of [NDArray]
     */
    fun create(data: BooleanArray): NDArray = create(data, Shape(data.size.toLong()))

    /**
     * Creates and initializes a 2D [NDArray].
     *
     * @param data the float array that needs to be set
     * @return a new instance of [NDArray]
     */
    fun create(data: Array<FloatArray>): NDArray {
        val buffer = FloatBuffer.allocate(data.size * data[0].size)
        for (d in data)
            buffer.put(d)
        buffer.rewind()
        return create(buffer, Shape(data.size.toLong(), data[0].size.toLong()))
    }

    /**
     * Creates and initializes a 2D [NDArray].
     *
     * @param data the float array that needs to be set
     * @return a new instance of [NDArray]
     */
    fun create(data: Array<IntArray>): NDArray {
        val buffer = IntBuffer.allocate(data.size * data[0].size)
        for (d in data) {
            buffer.put(d)
        }
        buffer.rewind()
        return create(buffer, Shape(data.size.toLong(), data[0].size.toLong()))
    }

    /**
     * Creates and initializes a 2D [NDArray].
     *
     * @param data the float array that needs to be set
     * @return a new instance of [NDArray]
     */
    fun create(data: Array<DoubleArray>): NDArray {
        val buffer = DoubleBuffer.allocate(data.size * data[0].size)
        for (d in data) {
            buffer.put(d)
        }
        buffer.rewind()
        return create(buffer, Shape(data.size.toLong(), data[0].size.toLong()))
    }

    /**
     * Creates and initializes a 2-D [NDArray].
     *
     * @param data the float array that needs to be set
     * @return a new instance of [NDArray]
     */
    fun create(data: Array<LongArray>): NDArray {
        val buffer = LongBuffer.allocate(data.size * data[0].size)
        for (d in data) {
            buffer.put(d)
        }
        buffer.rewind()
        return create(buffer, Shape(data.size.toLong(), data[0].size.toLong()))
    }

    /**
     * Creates and initializes a 2-D [NDArray].
     *
     * @param data the float array that needs to be set
     * @return a new instance of [NDArray]
     */
    fun create(data: Array<ByteArray>): NDArray {
        val buffer = ByteBuffer.allocate(data.size * data[0].size)
        for (d in data) {
            buffer.put(d)
        }
        buffer.rewind()
        return create(buffer, Shape(data.size.toLong(), data[0].size.toLong()))
    }

    /**
     * Creates and initializes a 2-D [NDArray].
     *
     * @param data the boolean array that needs to be set
     * @return a new instance of [NDArray]
     */
    fun create(data: Array<BooleanArray>): NDArray {
        val buffer = ByteBuffer.allocate(data.size * data[0].size)
        for (d in data) {
            for (b in d) {
                buffer.put((if (b) 1 else 0).toByte())
            }
        }
        buffer.rewind()
        return create(buffer, Shape(data.size.toLong(), data[0].size.toLong()), DataType.BOOLEAN)
    }

    /**
     * Creates and initializes a [NDArray] with specified [Shape].
     *
     *
     * [DataType] of the NDArray will determined by type of Buffer.
     *
     * @param data the data to initialize the `NDArray`
     * @param shape the [Shape] of the [NDArray]
     * @return a new instance of [NDArray]
     */
    fun create(data: Buffer, shape: Shape): NDArray {
        val dataType = DataType.fromBuffer(data)
        return create(data, shape, dataType)
    }

    /**
     * Creates an uninitialized instance of [NDArray] with specified [Shape], and [ ].
     *
     * @param shape the [Shape] of the [NDArray]
     * @param dataType the [DataType] of the [NDArray]
     * @return a new instance of [NDArray]
     */
    fun create(shape: Shape, dataType: DataType): NDArray

    /**
     * Creates and initializes an instance of [NDArray] with specified [Shape] and
     * [DataType].
     *
     * @param data the data to initialize the [NDArray]
     * @param shape the [Shape] of the [NDArray]
     * @param dataType the [DataType] of the [NDArray]
     * @return a new instance of [NDArray]
     */
    fun create(data: Buffer, shape: Shape, dataType: DataType): NDArray {
        val array = create(shape, dataType)
        array.setFrom(data)
        return array
    }

    /**
     * Creates and initializes a 1D [NDArray].
     *
     * @param data the float array that needs to be set
     * @return a new instance of [NDArray]
     */
    infix fun create(data: FloatArray) = create(data, Shape(data.size.toLong()))

    /**
     * Creates and initializes an instance of [NDArray] with specified [Shape] and float
     * array.
     *
     * @param data the float array that needs to be set
     * @param shape the [Shape] of the [NDArray]
     * @return a new instance of [NDArray]
     */
    fun create(data: FloatArray, shape: Shape = Shape(data.size.toLong())): NDArray =
        create(FloatBuffer.wrap(data), shape)

    /**
     * Creates and initializes a 1D [NDArray].
     *
     * @param data the float array that needs to be set
     * @return a new instance of [NDArray]
     */
    infix fun create(data: IntArray) = create(data, Shape(data.size.toLong()))

    /**
     * Creates and initializes an instance of [NDArray] with specified [Shape] and int
     * array.
     *
     * @param data the float array that needs to be set
     * @param shape the [Shape] of the [NDArray]
     * @return a new instance of [NDArray]
     */
    fun create(data: IntArray, shape: Shape): NDArray =
        create(IntBuffer.wrap(data), shape)

    /**
     * Creates and initializes a 1D [NDArray].
     *
     * @param data the float array that needs to be set
     * @return a new instance of [NDArray]
     */
    infix fun create(data: DoubleArray) = create(data, Shape(data.size.toLong()))

    /**
     * Creates and initializes an instance of [NDArray] with specified [Shape] and
     * double array.
     *
     * @param data the float array that needs to be set
     * @param shape the [Shape] of the [NDArray]
     * @return a new instance of [NDArray]
     */
    fun create(data: DoubleArray, shape: Shape = Shape(data.size.toLong())): NDArray =
        create(DoubleBuffer.wrap(data), shape)

    /**
     * Creates and initializes a 1D [NDArray].
     *
     * @param data the float array that needs to be set
     * @return a new instance of [NDArray]
     */
    infix fun create(data: LongArray) = create(data, Shape(data.size.toLong()))

    /**
     * Creates and initializes an instance of [NDArray] with specified [Shape] and long
     * array.
     *
     * @param data the float array that needs to be set
     * @param shape the [Shape] of the [NDArray]
     * @return a new instance of [NDArray]
     */
    fun create(data: LongArray, shape: Shape): NDArray =
        create(LongBuffer.wrap(data), shape)

    /**
     * Creates and initializes a 1D [NDArray].
     *
     * @param data the float array that needs to be set
     * @return a new instance of [NDArray]
     */
    infix fun create(data: ByteArray) = create(data, Shape(data.size.toLong()))

    /**
     * Creates and initializes an instance of [NDArray] with specified [Shape] and byte
     * array.
     *
     * @param data the float array that needs to be set
     * @param shape the [Shape] of the [NDArray]
     * @return a new instance of [NDArray]
     */
    fun create(data: ByteArray, shape: Shape): NDArray =
        create(ByteBuffer.wrap(data), shape)

    /**
     * Creates and initializes an instance of [NDArray] with specified [Shape] and
     * boolean array.
     *
     * @param data the boolean array that needs to be set
     * @param shape the [Shape] of the [NDArray]
     * @return a new instance of [NDArray]
     */
    fun create(data: BooleanArray, shape: Shape): NDArray {
        val byteData = ByteArray(data.size)
        for (i in data.indices)
            byteData[i] = (if (data[i]) 1 else 0).toByte()
        return create(ByteBuffer.wrap(byteData), shape, DataType.BOOLEAN)
    }

    /**
     * Creates an uninitialized instance of [DataType.FLOAT32] [NDArray] with specified
     * [Shape].
     *
     * @param shape the [Shape] of the [NDArray]
     * @return a new instance of [NDArray]
     */
    infix fun create(shape: Shape) = create(shape, DataType.FLOAT32, device)

    /**
     * Creates an uninitialized instance of [NDArray] with specified [Shape], [ ] and [Device].
     *
     * @param shape the [Shape] of the [NDArray]
     * @param dataType the [DataType] of the [NDArray]
     * @param device the [Device] of the [NDArray]
     * @return a new instance of [NDArray]
     */
    fun create(shape: Shape, dataType: DataType, device: Device?): NDArray {
        if (device == null || device == this.device)
            return create(shape, dataType)
        return newSubManager(device).create(shape, dataType)
    }

    /**
     * Creates a Compressed Sparse Row Storage (CSR) Format Matrix.
     *
     * @param data the data to set for the CSR Matrix
     * @param indptr the indptr array is what will help identify the rows where the data appears
     * @param indices the indices array stores the column index for each non-zero element in data
     * @param shape the [Shape] of the [NDArray]
     * @param device the [Device] of the [NDArray]
     * @return a new instance of [NDArray]
     */
    fun createCSR(data: FloatArray, indptr: LongArray, indices: LongArray, shape: Shape, device: Device?): NDArray =
        createCSR(FloatBuffer.wrap(data), indptr, indices, shape, device)

    /**
     * Creates a Compressed Sparse Row Storage (CSR) Format Matrix.
     *
     * @param data the data to set for the CSR Matrix
     * @param indptr the indptr array is what will help identify the rows where the data appears
     * @param indices the indices array stores the column index for each non-zero element in data
     * @param shape the [Shape] of the [NDArray]
     * @param device the [Device] of the [NDArray]
     * @return a new instance of [NDArray]
     */
    fun createCSR(data: Buffer, indptr: LongArray, indices: LongArray, shape: Shape, device: Device?): NDArray {
        if (device == null || device == this.device)
            return createCSR(data, indptr, indices, shape)
        return newSubManager(device).createCSR(data, indptr, indices, shape)
    }

    /**
     * Creates a Compressed Sparse Row Storage (CSR) Format Matrix.
     *
     * @param data the data to set for the CSR Matrix
     * @param indptr the indptr array is what will help identify the rows where the data appears
     * @param indices the indices array stores the column index for each non-zero element in data
     * @param shape the [Shape] of the [NDArray]
     * @return a new instance of [NDArray]
     */
    fun createCSR(data: Buffer, indptr: LongArray, indices: LongArray, shape: Shape): NDArray

    /**
     * Stores the matrix in row sparse format.
     *
     * @param data the data to set for the Row Sparse [NDArray]
     * @param dataShape the [Shape] of the data [NDArray]
     * @param indices the indices to store the data
     * @param shape the [Shape] of the [NDArray]
     * @param device the [Device] of the [NDArray]
     * @return a new instance of [NDArray]
     */
    fun createRowSparse(data: Buffer, dataShape: Shape, indices: LongArray, shape: Shape, device: Device?): NDArray {
        if (device == null || device == this.device)
            return createRowSparse(data, dataShape, indices, shape)
        return newSubManager(device).createRowSparse(data, dataShape, indices, shape)
    }

    /**
     * Stores the matrix in row sparse format.
     *
     * @param data the data to set for the Row Sparse [NDArray]
     * @param dataShape the [Shape] of the data [NDArray]
     * @param indices the indices to store the data
     * @param shape the [Shape] of the [NDArray]
     * @return a new instance of [NDArray]
     */
    fun createRowSparse(data: Buffer, dataShape: Shape, indices: LongArray, shape: Shape): NDArray

    /**
     * Creates a Coordinate Format (COO) Matrix.
     *
     * @param data the data to set for the Coordinate format [NDArray]
     * @param indices the matrix represent indices
     * @param shape the [Shape] of the [NDArray]
     * @return a new instance of [NDArray]
     */
    fun createCoo(data: Buffer, indices: Array<LongArray>, shape: Shape): NDArray

    /**
     * Decodes [NDArray] through byte array.
     *
     * @param bytes byte array to load from
     * @return [NDArray]
     */
    fun decode(bytes: ByteArray): NDArray = NDSerializer.decode(this, ByteBuffer.wrap(bytes))

    /**
     * Decodes [NDArray] through [DataInputStream].
     *
     * @param is input stream data to load from
     * @return [NDArray]
     * @throws IOException data is not readable
     */
    @Throws(IOException::class)
    fun decode(`is`: InputStream?): NDArray = NDSerializer.decode(this, `is`)

    /**
     * Loads the NDArrays saved to a file.
     *
     * @param path the path to the file
     * @return the loaded arrays
     */
    fun load(path: Path): NDList

    /**
     * Loads the NDArrays saved to a file.
     *
     * @param path the path to the file
     * @param device the device to use for the loaded arrays
     * @return the loaded arrays
     */
    fun load(path: Path, device: Device?): NDList {
        if (device == null || device == this.device)
            return load(path)
        return newSubManager(device).load(path)
    }

    /**
     * Gets the name of the NDManager.
     *
     * @return name
     */
    /**
     * Sets the name for the NDManager.
     *
     * @param name the name assigned to the manager
     */
    //    @JvmField
    var name: String

    /**
     * Creates an instance of [NDArray] with specified [Shape] filled with zeros.
     *
     * @param shape the [Shape] of the [NDArray]
     * @return a new instance of [NDArray]
     * @see .zeros
     */
    infix fun zeros(shape: Shape): NDArray = zeros(shape, DataType.FLOAT32)

    /**
     * Creates an instance of [NDArray] with specified [Shape] filled with zeros.
     *
     * @param shape the [Shape] of the [NDArray]
     * @param dataType the [DataType] of the [NDArray]
     * @return a new instance of [NDArray]
     * @see .zeros
     */
    fun zeros(shape: Shape, dataType: DataType): NDArray {
        val size = shape.size().toInt()
        val bb = allocateDirect(size * dataType.numOfBytes)
        return create(bb, shape, dataType)
    }

    /**
     * Creates an instance of [NDArray] with specified [Device], [Shape], and
     * [DataType] filled with zeros.
     *
     * @param shape the [Shape] of the [NDArray]
     * @param dataType the [DataType] of the [NDArray]
     * @param device the [Device] of the [NDArray]
     * @return a new instance of [NDArray]
     */
    fun zeros(shape: Shape, dataType: DataType, device: Device?): NDArray {
        if (device == null || device == this.device)
            return zeros(shape, dataType)
        return newSubManager(device).zeros(shape, dataType)
    }

    /**
     * Creates an instance of [NDArray] with specified [Shape] filled with ones.
     *
     * @param shape the [Shape] of the [NDArray]
     * @param dataType the [DataType] of the [NDArray]
     * @return a new instance of [NDArray]
     */
    fun ones(shape: Shape, dataType: DataType): NDArray {
        val size = shape.size().toInt()
        val bb = allocateDirect(size * dataType.numOfBytes)
        for (i in 0..<size)
            when (dataType) {
                DataType.FLOAT16 -> bb.putShort(Float16Utils.ONE)
                DataType.FLOAT32 -> bb.putFloat(1f)
                DataType.FLOAT64 -> bb.putDouble(1.0)
                DataType.INT32 -> bb.putInt(1)
                DataType.INT64 -> bb.putLong(1)
                DataType.UINT8, DataType.INT8 -> bb.put(1.toByte())
                DataType.UNKNOWN -> {}
                else -> {}
            }
        bb.rewind()
        return create(bb, shape, dataType)
    }

    /**
     * Creates an instance of [NDArray] with specified [Shape] filled with ones.
     *
     * @param shape the [Shape] of the [NDArray]
     * @return a new instance of [NDArray]
     */
    infix fun ones(shape: Shape): NDArray = ones(shape, DataType.FLOAT32)

    /**
     * Creates an instance of [NDArray] with specified [Device], [Shape], and
     * [DataType] filled with ones.
     *
     * @param shape the [Shape] of the [NDArray]
     * @param dataType the [DataType] of the [NDArray]
     * @param device the [Device] of the [NDArray]
     * @return a new instance of [NDArray]
     */
    fun ones(shape: Shape, dataType: DataType, device: Device?): NDArray {
        if (device == null || device == this.device)
            return ones(shape, dataType)
        return newSubManager(device).ones(shape, dataType)
    }

    /**
     * Return a new `NDArray` of given shape, filled with value.
     *
     * @param shape shape of a new `NDArray`
     * @param value fill value
     * @return `NDArray` of fill value with the given shape
     */
    fun full(shape: Shape, value: Int): NDArray = full(shape, value.toFloat(), DataType.INT32)

    /**
     * Return a new `NDArray` of given shape, filled with value.
     *
     * @param shape shape of a new `NDArray`
     * @param value fill value
     * @return `NDArray` of fill value with the given shape
     */
    fun full(shape: Shape, value: Float): NDArray = full(shape, value, DataType.FLOAT32)

    /**
     * Return a new `NDArray` of given shape, filled with value.
     *
     * @param shape shape of a new `NDArray`
     * @param value fill value
     * @param dataType the desired data-type for the [NDArray]
     * @return `NDArray` of fill value with the given shape
     */
    fun full(shape: Shape, value: Float, dataType: DataType): NDArray

    /**
     * Return a new `NDArray` of given shape, device, filled with value.
     *
     * @param shape shape of a new `NDArray`
     * @param value fill value
     * @param dataType the desired data-type for the [NDArray]
     * @param device the [Device] of the [NDArray]
     * @return `NDArray` of fill value with the given shape
     */
    fun full(shape: Shape, value: Float, dataType: DataType, device: Device?): NDArray {
        if (device == null || device == this.device)
            return full(shape, value, dataType)
        return newSubManager(device).full(shape, value, dataType)
    }

    /**
     * Returns evenly spaced values starting from 0.
     *
     *
     * Values are generated within the half-open interval [start, stop) (in other words, the
     * interval including start but excluding stop). For integer arguments, the function is
     * equivalent to the Python built-in range function, but returns an instance of [NDArray]
     * rather than a list.
     *
     * @param stop the end of the interval. The interval does not include this value
     * @return a new instance of [NDArray]
     */
    fun arange(stop: Int): NDArray = arange(0, stop, 1, DataType.INT32)

    /**
     * Returns evenly spaced values starting from 0.
     *
     *
     * Values are generated within the half-open interval [start, stop) (in other words, the
     * interval including start but excluding stop). For integer arguments, the function is
     * equivalent to the Python built-in range function, but returns an instance of [NDArray]
     * rather than a list.
     *
     * @param stop the end of the interval. The interval does not include this value
     * @return a new instance of [NDArray]
     */
    fun arange(stop: Float): NDArray = arange(0.0f, stop, 1.0f, DataType.FLOAT32)

    /**
     * Returns evenly spaced values within a given interval with step 1.
     *
     *
     * Values are generated within the half-open interval [start, stop) (in other words, the
     * interval including start but excluding stop). For integer arguments, the function is
     * equivalent to the Python built-in range function, but returns an instance of [NDArray]
     * rather than a list.
     *
     * @param start the start of interval. The interval includes this value
     * @param stop the end of interval. The interval does not include this value
     * @return a new instance of [NDArray]
     */
    fun arange(start: Int, stop: Int): NDArray = arange(start, stop, 1, DataType.INT32)

    /**
     * Returns evenly spaced values within a given interval with step 1.
     *
     *
     * Values are generated within the half-open interval [start, stop) (in other words, the
     * interval including start but excluding stop). For integer arguments, the function is
     * equivalent to the Python built-in range function, but returns an instance of [NDArray]
     * rather than a list.
     *
     * @param start the start of interval. The interval includes this value
     * @param stop the end of interval. The interval does not include this value
     * @return a new instance of [NDArray]
     */
    fun arange(start: Float, stop: Float): NDArray = arange(start, stop, 1.0f, DataType.FLOAT32)

    /**
     * Returns evenly spaced values within a given interval.
     *
     *
     * Values are generated within the half-open interval [start, stop) (in other words, the
     * interval including start but excluding stop). For integer arguments, the function is
     * equivalent to the Python built-in range function, but returns an instance of [NDArray]
     * rather than a list.
     *
     * @param start the start of interval. The interval includes this value
     * @param stop the end of interval. The interval does not include this value
     * @param step the spacing between values
     * @return a new instance of [NDArray]
     */
    fun arange(start: Int, stop: Int, step: Int): NDArray = arange(start, stop, step, DataType.INT32)

    /**
     * Returns evenly spaced values within a given interval.
     *
     *
     * Values are generated within the half-open interval [start, stop) (in other words, the
     * interval including start but excluding stop). For integer arguments, the function is
     * equivalent to the Python built-in range function, but returns an instance of [NDArray]
     * rather than a list.
     *
     * @param start the start of interval. The interval includes this value
     * @param stop the end of interval. The interval does not include this value
     * @param step the spacing between values
     * @return a new instance of [NDArray]
     */
    fun arange(start: Float, stop: Float, step: Float): NDArray = arange(start, stop, step, DataType.FLOAT32)

    /**
     * Returns evenly spaced values within a given interval.
     *
     *
     * Values are generated within the half-open interval [start, stop) (in other words, the
     * interval including start but excluding stop). For integer arguments, the function is
     * equivalent to the Python built-in range function, but returns an instance of [NDArray]
     * rather than a list.
     *
     * @param start the start of interval. The interval includes this value
     * @param stop the end of interval. The interval does not include this value
     * @param step the spacing between values
     * @param dataType the [DataType] of the [NDArray]
     * @return a new instance of [NDArray]
     */
    fun arange(start: Int, stop: Int, step: Int, dataType: DataType): NDArray =
        arange(start.toFloat(), stop.toFloat(), step.toFloat(), dataType)

    /**
     * Returns evenly spaced values within a given interval.
     *
     *
     * Values are generated within the half-open interval [start, stop) (in other words, the
     * interval including start but excluding stop). For integer arguments, the function is
     * equivalent to the Python built-in range function, but returns an instance of [NDArray]
     * rather than a list.
     *
     * @param start the start of interval. The interval includes this value
     * @param stop the end of interval. The interval does not include this value
     * @param step the spacing between values
     * @param dataType the [DataType] of the [NDArray]
     * @return a new instance of [NDArray]
     */
    fun arange(start: Float, stop: Float, step: Float, dataType: DataType): NDArray

    /**
     * Returns evenly spaced values within a given interval.
     *
     *
     * Values are generated within the half-open interval [start, stop) (in other words, the
     * interval including start but excluding stop). For integer arguments, the function is
     * equivalent to the Python built-in range function, but returns an instance of [NDArray]
     * rather than a list.
     *
     * @param start the start of interval. The interval includes this value
     * @param stop the end of interval. The interval does not include this value
     * @param step the spacing between values
     * @param dataType the [DataType] of the [NDArray]
     * @param device the [Device] of the [NDArray]
     * @return a new instance of [NDArray]
     */
    fun arange(start: Float, stop: Float, step: Float, dataType: DataType, device: Device?): NDArray {
        if (device == null || device == this.device)
            return arange(start, stop, step, dataType)
        return newSubManager(device).arange(start, stop, step, dataType)
    }

    /**
     * Returns a 2-D array with ones on the diagonal and zeros elsewhere.
     *
     * @param rows the number of rows and cols in the output
     * @return a [NDArray] where all elements are equal to zero, except for the k-th diagonal,
     * whose values are equal to one
     */
    fun eye(rows: Int): NDArray = eye(rows, rows, 0, DataType.FLOAT32)

    /**
     * Returns a 2-D array with ones on the diagonal and zeros elsewhere.
     *
     * @param rows the number of rows and cols in the output
     * @param k the index of the diagonal: a positive value refers to an upper diagonal, and a
     * negative value to a lower diagonal
     * @return a [NDArray] where all elements are equal to zero, except for the k-th diagonal,
     * whose values are equal to one
     */
    fun eye(rows: Int, k: Int): NDArray = eye(rows, rows, k, DataType.FLOAT32)

    /**
     * Returns a 2-D array with ones on the diagonal and zeros elsewhere.
     *
     * @param rows the number of rows in the output
     * @param cols the number of columns in the output
     * @param k the index of the diagonal: a positive value refers to an upper diagonal, and a
     * negative value to a lower diagonal
     * @return a [NDArray] where all elements are equal to zero, except for the k-th diagonal,
     * whose values are equal to one
     */
    fun eye(rows: Int, cols: Int, k: Int): NDArray = eye(rows, cols, k, DataType.FLOAT32)

    /**
     * Returns a 2-D array with ones on the diagonal and zeros elsewhere.
     *
     * @param rows the number of rows int the output
     * @param cols the number of columns in the output
     * @param k the index of the diagonal: a positive value refers to an upper diagonal, and a
     * negative value to a lower diagonal
     * @param dataType the [DataType] of the [NDArray]
     * @return a [NDArray] where all elements are equal to zero, except for the k-th diagonal,
     * whose values are equal to one
     */
    fun eye(rows: Int, cols: Int, k: Int, dataType: DataType): NDArray

    /**
     * Returns a 2-D array with ones on the diagonal and zeros elsewhere.
     *
     * @param rows the number of rows int the output
     * @param cols the number of columns in the output
     * @param k the index of the diagonal: a positive value refers to an upper diagonal, and a
     * negative value to a lower diagonal
     * @param dataType the [DataType] of the [NDArray]
     * @param device the [Device] of the [NDArray]
     * @return a [NDArray] where all elements are equal to zero, except for the k-th diagonal,
     * whose values are equal to one
     */
    fun eye(rows: Int, cols: Int, k: Int, dataType: DataType, device: Device?): NDArray {
        if (device == null || device == this.device)
            return eye(rows, cols, k, dataType)
        return newSubManager(device).eye(rows, cols, k, dataType)
    }

    /**
     * Returns evenly spaced numbers over a specified interval.
     *
     *
     * Returns num evenly spaced samples, calculated over the interval [start, stop].
     *
     * @param start the starting value of the sequence
     * @param stop the end value of the sequence
     * @param num the number of samples to generate
     * @return a new instance of [NDArray]
     */
    fun linspace(start: Float, stop: Float, num: Int): NDArray = linspace(start, stop, num, true)

    /**
     * Returns evenly spaced numbers over a specified interval.
     *
     *
     * Returns num evenly spaced samples, calculated over the interval [start, stop].The endpoint
     * of the interval can optionally be excluded.
     *
     * @param start the starting value of the sequence
     * @param stop the end value of the sequence
     * @param num the number of samples to generate
     * @param endpoint if `true`, stop is the last sample, otherwise, it is not included
     * @return a new instance of [NDArray]
     */
    /**
     * Returns evenly spaced numbers over a specified interval.
     *
     *
     * Returns num evenly spaced samples, calculated over the interval [start, stop].
     *
     * @param start the starting value of the sequence
     * @param stop the end value of the sequence
     * @param num the number of samples to generate
     * @return a new instance of [NDArray]
     */
    //    @JvmOverloads
    fun linspace(start: Int, stop: Int, num: Int, endpoint: Boolean = true): NDArray =
        linspace(start.toFloat(), stop.toFloat(), num, endpoint)

    /**
     * Returns evenly spaced numbers over a specified interval.
     *
     *
     * Returns num evenly spaced samples, calculated over the interval [start, stop].The endpoint
     * of the interval can optionally be excluded.
     *
     * @param start the starting value of the sequence
     * @param stop the end value of the sequence
     * @param num the number of samples to generate
     * @param endpoint if `true`, stop is the last sample, otherwise, it is not included
     * @return a new instance of [NDArray]
     */
    fun linspace(start: Float, stop: Float, num: Int, endpoint: Boolean): NDArray

    /**
     * Returns evenly spaced numbers over a specified interval.
     *
     *
     * Returns num evenly spaced samples, calculated over the interval [start, stop].The endpoint
     * of the interval can optionally be excluded.
     *
     * @param start the starting value of the sequence
     * @param stop the end value of the sequence
     * @param num the number of samples to generate
     * @param endpoint if `true`, stop is the last sample, otherwise, it is not included
     * @param device the [Device] of the [NDArray]
     * @return a new instance of [NDArray]
     */
    fun linspace(start: Float, stop: Float, num: Int, endpoint: Boolean, device: Device?): NDArray {
        if (device == null || device == this.device)
            return linspace(start, stop, num, endpoint)
        return newSubManager(device).linspace(start, stop, num, endpoint)
    }

    /**
     * Returns random integer values from low (inclusive) to high (exclusive).
     *
     * @param low Lowest (signed) longs to be drawn from the distribution
     * @param high one above the largest (signed) long to be drawn from the distribution
     * @param shape the [Shape] of the [NDArray]
     * @param dataType the [DataType] of the [NDArray]
     * @return the drawn samples [NDArray]
     */
    fun randomInteger(low: Long, high: Long, shape: Shape, dataType: DataType): NDArray

    /**
     * Returns a random permutation of integers from 0 to n - 1.
     *
     * @param n (int) – the upper bound (exclusive)
     * @return a random permutation of integers from 0 to n - 1.
     */
    fun randomPermutation(n: Long): NDArray

    /**
     * Draws samples from a uniform distribution.
     *
     *
     * Samples are uniformly distributed over the half-open interval [low, high) (includes low,
     * but excludes high). In other words, any value within the given interval is equally likely to
     * be drawn by uniform.
     *
     * @param low the lower boundary of the output interval. All values generated will be greater
     * than or equal to low.
     * @param high the upper boundary of the output interval. All values generated will be less than
     * high.
     * @param shape the [Shape] of the [NDArray]
     * @return the drawn samples [NDArray]
     */
    fun randomUniform(low: Float, high: Float, shape: Shape): NDArray =
        randomUniform(low, high, shape, DataType.FLOAT32)

    /**
     * Draws samples from a uniform distribution.
     *
     *
     * Samples are uniformly distributed over the half-open interval [low, high) (includes low,
     * but excludes high). In other words, any value within the given interval is equally likely to
     * be drawn by uniform.
     *
     * @param low the lower boundary of the output interval. All values generated will be greater
     * than or equal to low.
     * @param high the upper boundary of the output interval. All values generated will be less than
     * high.
     * @param shape the [Shape] of the [NDArray]
     * @param dataType the [DataType] of the [NDArray]
     * @return the drawn samples [NDArray]
     */
    fun randomUniform(low: Float, high: Float, shape: Shape, dataType: DataType): NDArray

    /**
     * Draws samples from a uniform distribution.
     *
     *
     * Samples are uniformly distributed over the half-open interval [low, high) (includes low,
     * but excludes high). In other words, any value within the given interval is equally likely to
     * be drawn by uniform.
     *
     * @param low the lower boundary of the output interval. All values generated will be greater
     * than or equal to low.
     * @param high the upper boundary of the output interval. All values generated will be less than
     * high.
     * @param shape the [Shape] of the [NDArray]
     * @param dataType the [DataType] of the [NDArray]
     * @param device the [Device] of the [NDArray]
     * @return the drawn samples [NDArray]
     */
    fun randomUniform(low: Float, high: Float, shape: Shape, dataType: DataType, device: Device?): NDArray {
        if (device == null || device == this.device)
            return randomUniform(low, high, shape, dataType)
        return newSubManager(device).randomUniform(low, high, shape, dataType)
    }

    /**
     * Draws random samples from a normal (Gaussian) distribution with mean 0 and standard deviation
     * 1.
     *
     *
     * Samples are distributed according to a normal distribution parametrized by mean = 0 and
     * standard deviation = 1.
     *
     * @param shape the output [Shape]
     * @return the drawn samples [NDArray]
     */
    fun randomNormal(shape: Shape): NDArray = randomNormal(0f, 1f, shape, DataType.FLOAT32)

    /**
     * Draws random samples from a normal (Gaussian) distribution with mean 0 and standard deviation
     * 1.
     *
     * @param shape the output [Shape]
     * @param dataType the [DataType] of the [NDArray]
     * @return the drawn samples [NDArray]
     */
    fun randomNormal(shape: Shape, dataType: DataType): NDArray = randomNormal(0.0f, 1.0f, shape, dataType)

    /**
     * Draws random samples from a normal (Gaussian) distribution.
     *
     * @param loc the mean (centre) of the distribution
     * @param scale the standard deviation (spread or "width") of the distribution
     * @param shape the output [Shape]
     * @param dataType the [DataType] of the [NDArray]
     * @return the drawn samples [NDArray]
     */
    fun randomNormal(loc: Float, scale: Float, shape: Shape, dataType: DataType): NDArray

    /**
     * Draws random samples from a normal (Gaussian) distribution.
     *
     * @param loc the mean (centre) of the distribution
     * @param scale the standard deviation (spread or "width") of the distribution
     * @param shape the output [Shape]
     * @param dataType the [DataType] of the [NDArray]
     * @param device the [Device] of the [NDArray]
     * @return the drawn samples [NDArray]
     */
    fun randomNormal(loc: Float, scale: Float, shape: Shape, dataType: DataType, device: Device?): NDArray {
        if (device == null || device == this.device)
            return randomNormal(loc, scale, shape, dataType)
        return newSubManager(device).randomNormal(loc, scale, shape, dataType)
    }

    /**
     * Draws random samples from a normal (Gaussian) distribution with mean 0 and standard deviation
     * 1, discarding and re-drawing any samples that are more than two standard deviations from the
     * mean.
     *
     *
     * Samples are distributed according to a normal distribution parametrized by mean = 0 and
     * standard deviation = 1.
     *
     * @param shape the output [Shape]
     * @return the drawn samples [NDArray]
     */
    fun truncatedNormal(shape: Shape): NDArray = truncatedNormal(0f, 1f, shape, DataType.FLOAT32)

    /**
     * Draws random samples from a normal (Gaussian) distribution with mean 0 and standard deviation
     * 1, discarding and re-drawing any samples that are more than two standard deviations from the
     * mean.
     *
     * @param shape the output [Shape]
     * @param dataType the [DataType] of the [NDArray]
     * @return the drawn samples [NDArray]
     */
    fun truncatedNormal(shape: Shape, dataType: DataType): NDArray = truncatedNormal(0.0f, 1.0f, shape, dataType)

    /**
     * Draws random samples from a normal (Gaussian) distribution, discarding and re-drawing any
     * samples that are more than two standard deviations from the mean.
     *
     * @param loc the mean (centre) of the distribution
     * @param scale the standard deviation (spread or "width") of the distribution
     * @param shape the output [Shape]
     * @param dataType the [DataType] of the [NDArray]
     * @return the drawn samples [NDArray]
     */
    fun truncatedNormal(loc: Float, scale: Float, shape: Shape, dataType: DataType): NDArray

    /**
     * Draws random samples from a normal (Gaussian) distribution, discarding and re-drawing any
     * samples that are more than two standard deviations from the mean.
     *
     * @param loc the mean (centre) of the distribution
     * @param scale the standard deviation (spread or "width") of the distribution
     * @param shape the output [Shape]
     * @param dataType the [DataType] of the [NDArray]
     * @param device the [Device] of the [NDArray]
     * @return the drawn samples [NDArray]
     */
    fun truncatedNormal(
        loc: Float, scale: Float, shape: Shape, dataType: DataType, device: Device?): NDArray {
        if (device == null || device == this.device)
            return truncatedNormal(loc, scale, shape, dataType)
        return newSubManager(device).truncatedNormal(loc, scale, shape, dataType)
    }

    /**
     * Draw samples from a multinomial distribution.
     *
     *
     * The multinomial distribution is a multivariate generalization of the binomial
     * distribution. Take an experiment with one of p possible outcomes. An example of such an
     * experiment is throwing a dice, where the outcome can be 1 through 6. Each sample drawn from
     * the distribution represents n such experiments. Its values, X_i = [X_0, X_1, ..., X_p],
     * represent the number of times the outcome was i.
     *
     * @param n the number of experiments
     * @param pValues the probabilities of each of the p different outcomes. These should sum to 1
     * The last element is always assumed to account for the remaining probability, as long as
     * pValues.sum().getFloat() &lt;= 1)
     * @return the drawn samples [NDArray]
     */
    fun randomMultinomial(n: Int, pValues: NDArray): NDArray

    /**
     * Draw samples from a multinomial distribution.
     *
     *
     * The multinomial distribution is a multivariate generalization of the binomial
     * distribution. Take an experiment with one of p possible outcomes. An example of such an
     * experiment is throwing a dice, where the outcome can be 1 through 6. Each sample drawn from
     * the distribution represents n such experiments. Its values, X_i = [X_0, X_1, ..., X_p],
     * represent the number of times the outcome was i.
     *
     * @param n the number of experiments
     * @param pValues the probabilities of each of the p different outcomes. These should sum to 1
     * The last element is always assumed to account for the remaining probability, as long as
     * pValues.sum().getFloat() &lt;= 1)
     * @param shape the output [Shape]
     * @return the drawn samples [NDArray]
     */
    fun randomMultinomial(n: Int, pValues: NDArray, shape: Shape): NDArray

    /**
     * Concurrent sampling from multiple normal distributions with parameters *mu* (mean) and
     * *sigma* (standard deviation).
     *
     * @param mu Means of the distributions
     * @param sigma Standard deviations of the distributions
     * @return the drawn samples [NDArray]
     */
    fun sampleNormal(mu: NDArray, sigma: NDArray): NDArray

    /**
     * Concurrent sampling from multiple normal distributions with parameters *mu* (mean) and
     * *sigma* (standard deviation).
     *
     * @param mu Means of the distributions
     * @param sigma Standard deviations of the distributions
     * @param shape Shape to be sampled from each random distribution
     * @return the drawn samples [NDArray]
     */
    fun sampleNormal(mu: NDArray, sigma: NDArray, shape: Shape): NDArray

    /**
     * Draw random samples from a Poisson distribution.
     *
     *
     * Samples are distributed according to a Poisson distribution parametrized by *lambda*
     * (rate). Samples will always be returned as a floating point data type.
     *
     * @param lam Lambda (rate) parameters of the distributions
     * @return the drawn samples [NDArray]
     */
    fun samplePoisson(lam: NDArray): NDArray

    /**
     * Draw random samples from a Poisson distribution.
     *
     *
     * Samples are distributed according to a Poisson distribution parametrized by *lambda*
     * (rate). Samples will always be returned as a floating point data type.
     *
     * @param lam Lambda (rate) parameters of the distributions
     * @param shape Shape to be sampled from each random distribution
     * @return the drawn samples [NDArray]
     */
    fun samplePoisson(lam: NDArray, shape: Shape): NDArray

    /**
     * Draw random samples from a gamma distribution.
     *
     *
     * Samples are distributed according to a gamma distribution parametrized by *alpha* (shape)
     * and *beta* (scale).
     *
     * @param alpha The shape of the gamma distribution
     * @param beta The scale of the gamma distribution
     * @return the drawn samples [NDArray]
     */
    fun sampleGamma(alpha: NDArray, beta: NDArray): NDArray

    /**
     * Draw random samples from a gamma distribution.
     *
     *
     * Samples are distributed according to a gamma distribution parametrized by *alpha* (shape)
     * and *beta* (scale).
     *
     * @param alpha The shape of the gamma distribution
     * @param beta The scale of the gamma distribution
     * @param shape Shape to be sampled from each random distribution
     * @return the drawn samples [NDArray]
     */
    fun sampleGamma(alpha: NDArray, beta: NDArray, shape: Shape): NDArray

    /**
     * Builds the Hanning Window.
     *
     *
     * The Hanning was named for Julius von Hann, an Austrian meteorologist. It is also known as
     * the Cosine Bell. Some authors prefer that it be called a Hann window, to help avoid confusion
     * with the very similar Hamming window.
     *
     * @param numPoints Number of points in the output window.
     * @return the window
     */
    fun hanningWindow(numPoints: Long): NDArray {
        val data = FloatArray(numPoints.toInt())
        // shift from N -1 to N to trims off the last duplicate value from the symmetric window
        for (i in 1 until data.size)
            data[i] = (0.5 * (1 - cos((2 * Math.PI * i) / numPoints))).toFloat()
        return create(data)
    }

    /**
     * Check if the manager is still valid.
     *
     * @return the current status
     */
    //    @JvmField
    val isOpen: Boolean

    /**
     * Caps this manager to prevent unintentional attachment of resources. This is useful to detect
     * memory leaks at an early point in time. The attachment of sub managers is still allowed after
     * this method has been called.
     */
    fun cap()

    /**
     * Returns the parent `NDManager`.
     *
     * @return the parent `NDManager`
     */
    //    @JvmField
    val parentManager: NDManager?

    /**
     * Creates a child `NDManager`.
     *
     *
     * Child `NDManager` will inherit default [Device] from this `NDManager`.
     *
     * @return a child `NDManager`
     */
    fun newSubManager(): NDManager

    /**
     * Creates a child `NDManager` with specified default [Device].
     *
     * @param device the default [Device]
     * @return a child `NDManager`
     */
    fun newSubManager(device: Device): NDManager

    /**
     * Returns the default [Device] of this `NDManager`.
     *
     * @return the default [Device] of this `NDManager`
     */
    //    @JvmField
    val device: Device

    /**
     * Returns all [NDArray]s managed by this manager (including recursively).
     *
     * @return all [NDArray]s managed by this manager (including recursively)
     */
    //    @JvmField
    val managedArrays: List<NDArray>

    /**
     * Attaches a resource to this `NDManager`.
     *
     *
     * The attached resource will be closed when this `NDManager` is closed.
     *
     *
     * This attachment is internal. Many resources will internally track which manager they are
     * attached to. In that case, you should call [NDResource.attach] instead and
     * that should then call attachInternal.
     *
     * @param resourceId the unique resourceId
     * @param resource the [AutoCloseable] resource to be attached
     */
    fun attachInternal(resourceId: String?, vararg resource: AutoCloseable)

    /**
     * Attaches a resource to this `NDManager` circumventing any cap protection.
     *
     *
     * The attached resource will be closed when this `NDManager` is closed.
     *
     *
     * This attachment is internal. Many resources will internally track which manager they are
     * attached to. In that case, you should call [NDResource.attach] instead and
     * that should then call attachInternal.
     *
     * @param resourceId the unique resourceId
     * @param resource the [AutoCloseable] resource to be attached
     */
    fun attachUncappedInternal(resourceId: String?, resource: AutoCloseable?)

    /**
     * Temporarily attaches a resource to this `NDManager` to be returned when this is closed.
     *
     *
     * The attached resource will be returned to it's original manager when this `NDManager` is closed.
     *
     *
     * This attachment is internal. Many resources will internally track which manager they are
     * attached to. In that case, you should call [NDResource.attach] instead and
     * that should then call tempAttachInternal.
     *
     * @param originalManager the original manager to return the resource to
     * @param resourceId the unique resourceId
     * @param resource the [AutoCloseable] resource to be attached
     */
    fun tempAttachInternal(originalManager: NDManager?, resourceId: String?, resource: NDResource?)

    /**
     * Detaches a [NDArray] from this `NDManager`'s lifecycle.
     *
     *
     * The detached [NDArray] become un-managed, it's user's responsibility to close the
     * resource. Failed to close the resource has to wait on GC to be freed, and might cause out of
     * native memory.
     *
     *
     * This detach is internal. Many resources will internally track which manager they are
     * attached to. In that case, you should call [NDResource.detach] instead and that
     * should then call detachInternal.
     *
     * @param resourceId the resourceId to be removed from this `NDManager`'s lifecycle
     */
    fun detachInternal(resourceId: String?)

    /**
     * Returns a value outside of this manager by attaching to this manager's parent.
     *
     * @param resource the resource to return
     * @param <T> the type of the resource
     * @return the passed in resource, after attaching to a new manager
    </T> */
    fun <T : NDResource?> ret(resource: T): T {
        resource!!.attach(parentManager)
        return resource
    }

    /**
     * Attaches all resources to this manager.
     *
     * @param resources the resources to attach
     * @see NDResource.attach
     */
    fun attachAll(vararg resources: NDResource) {
        for (resource in resources) {
            resource.attach(this)
        }
    }

    /**
     * Temporarily attaches all resources to this manager.
     *
     * @param resources the resources to attach
     * @see NDResource.tempAttach
     */
    fun tempAttachAll(vararg resources: NDResource) {
        for (resource in resources) {
            resource.tempAttach(this)
        }
    }

    /**
     * An engine specific generic invocation to native operation.
     *
     *
     * You should avoid using this function if possible. Since this function is engine specific,
     * using this API may cause a portability issue. Native operation may not be compatible between
     * each version.
     *
     * @param operation the native operation to perform
     * @param src the [NDList] of source [NDArray]
     * @param dest the [NDList] to save output to
     * @param params the parameters to be passed to the native operation
     * @throws IllegalArgumentException if operation is not supported by Engine
     * @throws EngineException if operation failed in native engine
     */
    fun invoke(operation: String, src: Array<NDArray>, dest: Array<NDArray>, params: PairList<String, *>)

    /**
     * An engine specific generic invocation to native operation.
     *
     *
     * You should avoid using this function if possible. Since this function is engine specific,
     * using this API may cause a portability issue. Native operation may not compatible between
     * each version.
     *
     * @param operation the native operation to perform
     * @param src the [NDList] of source [NDArray]
     * @param params the parameters to be passed to the native operation
     * @return the output array of [NDArray]
     * @throws IllegalArgumentException if operation is not supported by Engine
     * @throws EngineException if operation failed in native engine
     */
    fun invoke(operation: String, src: NDList, params: PairList<String, *>): NDList

    /**
     * Returns the [Engine] associated with this manager.
     *
     * @return the [Engine] associated with this manager
     */
    //    @JvmField
    val engine: Engine?

    /** {@inheritDoc}  */
    override fun close()

    /**
     * A [SystemNDManager] is a marker class for a base NDManager.
     *
     *
     * Unlike a typical [NDManager], they can not be closed and don't track memory.
     */
    interface SystemNDManager

    companion object {
        /**
         * Creates a new top-level `NDManager`.
         *
         *
         * `NDManager` will inherit default [Device].
         *
         * @return a new top-level `NDManager`
         */
        @JvmStatic
        fun newBaseManager(): NDManager {
            if (Engine.getAllEngines().isEmpty())
                return PassthroughNDManager.INSTANCE
            return Engine.getInstance().newBaseManager()
        }

        /**
         * Creates a new top-level `NDManager` with specified [Device].
         *
         * @param device the default [Device]
         * @return a new top-level `NDManager`
         */
        @JvmStatic
        fun newBaseManager(device: Device?): NDManager = Engine.getInstance().newBaseManager(device)

        /**
         * Creates a new top-level `NDManager` with specified engine.
         *
         * @param engineName the name of the engine
         * @return a new top-level `NDManager`
         */
        @JvmStatic
        fun newBaseManager(engineName: String?): NDManager = Engine.getEngine(engineName).newBaseManager()

        /**
         * Creates a new top-level `NDManager` with specified [Device] and engine.
         *
         * @param device the default [Device]
         * @param engineName the name of the engine
         * @return a new top-level `NDManager`
         */
        @JvmStatic
        fun newBaseManager(device: Device?, engineName: String?): NDManager =
            Engine.getEngine(engineName).newBaseManager(device)

        /**
         * Creates a new manager based on the given resource.
         *
         * @param resource the resource to use
         * @return a new memory scrope containing the array
         */
        @JvmStatic
        fun subManagerOf(resource: NDResource): NDManager = resource.manager.newSubManager()
    }
}

/**
 * Draws random samples from a normal (Gaussian) distribution.
 *
 * @param loc the mean (centre) of the distribution
 * @param scale the standard deviation (spread or "width") of the distribution
 * @param shape the output [Shape]
 * @return the drawn samples [NDArray]
 */
inline fun NDManager.randomNormal(loc: Float, scale: Float, shape: Shape): NDArray = randomNormal(loc, scale, shape, DataType.FLOAT32)
