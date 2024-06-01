///*
// * Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.
// *
// * Licensed under the Apache License, Version 2.0 (the "License"). You may not use this file except in compliance
// * with the License. A copy of the License is located at
// *
// * http://aws.amazon.com/apache2.0/
// *
// * or in the "license" file accompanying this file. This file is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
// * OR CONDITIONS OF ANY KIND, either express or implied. See the License for the specific language governing permissions
// * and limitations under the License.
// */
package ai.djl.util.passthrough
//
//import ai.djl.Device
//import ai.djl.engine.Engine
//import ai.djl.ndarray.*
//import ai.djl.ndarray.types.DataType
//import ai.djl.ndarray.types.Shape
//import ai.djl.util.PairList
//import java.nio.Buffer
//import java.nio.ByteBuffer
//import java.nio.ByteOrder
//import java.nio.charset.Charset
//import java.nio.file.Path
//
///** An [NDManager] that does nothing, for use in extensions and hybrid engines.  */
//class PassthroughNDManager : NDManager {
//    override var engine: Engine? = null
//    override var device: Device
//
//    /**
//     * Constructs a new `PassthroughNDManager` instance.
//     *
//     * @param engine the [Engine] associated with this manager
//     * @param device the default [Device]
//     */
//    constructor(engine: Engine, device: Device?) {
//        this.engine = engine
//        this.device = device ?: engine.defaultDevice()
//    }
//
//    private constructor() {
//        device = Device.cpu()
//    }
//
//    /** {@inheritDoc}  */
//    override fun defaultDevice(): Device {
//        if (engine != null) {
//            return engine!!.defaultDevice()
//        }
//        return device
//    }
//
//    /** {@inheritDoc}  */
//    override fun allocateDirect(capacity: Int): ByteBuffer {
//        return ByteBuffer.allocateDirect(capacity).order(ByteOrder.nativeOrder())
//    }
//
//    /** {@inheritDoc}  */
//    override fun from(array: NDArray?): NDArray {
//        if (array == null || array is PassthroughNDArray)
//            return array!!
//        return create(array.toByteBuffer(), array.shape, array.dataType)
//    }
//
//    /**
//     * Creates a new [PassthroughNDArray].
//     *
//     * @param object the object to store
//     * @return a new `PassthroughNDArray`
//     */
//    fun create(`object`: Any?): PassthroughNDArray = PassthroughNDArray(this, `object`)
//
//    /** {@inheritDoc}  */
//    override fun create(data: Buffer, shape: Shape, dataType: DataType): NDArray {
//        val size = Math.toIntExact(shape.size())
//        BaseNDManager.validateBuffer(data, dataType, size)
//        if (data is ByteBuffer)
//            return PassthroughNDArray(this, data, shape, dataType)
//        val bb = ByteBuffer.allocate(size * dataType.numOfBytes)
//        bb.order(ByteOrder.nativeOrder())
//        BaseNDManager.copyBuffer(data, bb)
//        return PassthroughNDArray(this, bb, shape, dataType)
//    }
//
//    /** {@inheritDoc}  */
//    override fun create(data: Array<String>, charset: Charset, shape: Shape): NDArray = UNSUPPORTED()
//
//    /** {@inheritDoc}  */
//    override fun create(shape: Shape, dataType: DataType): NDArray = UNSUPPORTED()
//
//    /** {@inheritDoc}  */
//    override fun createCSR(data: Buffer, indptr: LongArray, indices: LongArray, shape: Shape): NDArray = UNSUPPORTED()
//
//    /** {@inheritDoc}  */
//    override fun createRowSparse(data: Buffer, dataShape: Shape, indices: LongArray, shape: Shape): NDArray = UNSUPPORTED()
//
//    /** {@inheritDoc}  */
//    override fun createCoo(data: Buffer, indices: Array<LongArray>, shape: Shape): NDArray = UNSUPPORTED()
//
//    /** {@inheritDoc}  */
//    override fun load(path: Path): NDList = UNSUPPORTED()
//
//    override var name = "PassthroughNDManager"
//
//    /** {@inheritDoc}  */
//    override fun full(shape: Shape, value: Float, dataType: DataType): NDArray = UNSUPPORTED()
//
//    /** {@inheritDoc}  */
//    override fun arange(start: Float, stop: Float, step: Float, dataType: DataType): NDArray = UNSUPPORTED()
//
//    /** {@inheritDoc}  */
//    override fun eye(rows: Int, cols: Int, k: Int, dataType: DataType): NDArray = UNSUPPORTED()
//
//    /** {@inheritDoc}  */
//    override fun linspace(start: Float, stop: Float, num: Int, endpoint: Boolean): NDArray = UNSUPPORTED()
//
//    /** {@inheritDoc}  */
//    override fun randomInteger(low: Long, high: Long, shape: Shape, dataType: DataType): NDArray = UNSUPPORTED()
//
//    /** {@inheritDoc}  */
//    override fun randomPermutation(n: Long): NDArray = UNSUPPORTED("Not supported!")
//
//    /** {@inheritDoc}  */
//    override fun randomUniform(low: Float, high: Float, shape: Shape, dataType: DataType): NDArray = UNSUPPORTED()
//
//    /** {@inheritDoc}  */
//    override fun randomNormal(loc: Float, scale: Float, shape: Shape, dataType: DataType): NDArray = UNSUPPORTED()
//
//    /** {@inheritDoc}  */
//    override fun truncatedNormal(loc: Float, scale: Float, shape: Shape, dataType: DataType): NDArray = UNSUPPORTED()
//
//    /** {@inheritDoc}  */
//    override fun randomMultinomial(n: Int, pValues: NDArray): NDArray = UNSUPPORTED()
//
//    /** {@inheritDoc}  */
//    override fun randomMultinomial(n: Int, pValues: NDArray, shape: Shape): NDArray = UNSUPPORTED()
//
//    /** {@inheritDoc}  */
//    override fun sampleNormal(mu: NDArray, sigma: NDArray): NDArray = UNSUPPORTED()
//
//    /** {@inheritDoc}  */
//    override fun sampleNormal(mu: NDArray, sigma: NDArray, shape: Shape): NDArray = UNSUPPORTED()
//
//    /** {@inheritDoc}  */
//    override fun samplePoisson(lam: NDArray): NDArray = UNSUPPORTED()
//
//    /** {@inheritDoc}  */
//    override fun samplePoisson(lam: NDArray, shape: Shape): NDArray = UNSUPPORTED()
//
//    /** {@inheritDoc}  */
//    override fun sampleGamma(alpha: NDArray, beta: NDArray): NDArray = UNSUPPORTED()
//
//    /** {@inheritDoc}  */
//    override fun sampleGamma(alpha: NDArray, beta: NDArray, shape: Shape): NDArray = UNSUPPORTED()
//
//    /** {@inheritDoc}  */
//    override var isOpen = true
//
//    /** {@inheritDoc}  */
//    override fun cap() {}
//
//    /** {@inheritDoc}  */
//    override var parentManager: NDManager = this
//
//    /** {@inheritDoc}  */
//    override fun newSubManager(): NDManager = this
//
//    /** {@inheritDoc}  */
//    override fun newSubManager(device: Device): NDManager {
//        return PassthroughNDManager(engine!!, device)
//    }
//
//    /** {@inheritDoc}  */
//    override val managedArrays: List<NDArray> = emptyList()
//
//    /** {@inheritDoc}  */
//    override fun attachInternal(resourceId: String?, vararg resource: AutoCloseable) {}
//
//    /** {@inheritDoc}  */
//    override fun attachUncappedInternal(resourceId: String?, resource: AutoCloseable?) {}
//
//    /** {@inheritDoc}  */
//    override fun tempAttachInternal(originalManager: NDManager?, resourceId: String?, resource: NDResource?) {}
//
//    /** {@inheritDoc}  */
//    override fun detachInternal(resourceId: String?) {}
//
//    /** {@inheritDoc}  */
//    override fun invoke(operation: String, src: Array<NDArray>, dest: Array<NDArray>, params: PairList<String, *>) = UNSUPPORTED()
//
//    /** {@inheritDoc}  */
//    override fun invoke(operation: String, src: NDList, params: PairList<String, *>): NDList = UNSUPPORTED()
//
//    /** {@inheritDoc}  */
//    override fun close() {}
//
//    companion object {
//        private fun UNSUPPORTED(message: String = "Not supported by PassthroughNDManager"): Nothing = throw UnsupportedOperationException(message)
//
//        @JvmField
//        val INSTANCE: PassthroughNDManager = PassthroughNDManager()
//    }
//}
