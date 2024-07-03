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
package ai.djl.engine

import ai.djl.Device
import ai.djl.Model
import ai.djl.ndarray.NDManager
import ai.djl.nn.SymbolBlock
import ai.djl.training.GradientCollector
import ai.djl.training.LocalParameterServer
import ai.djl.training.ParameterServer
import ai.djl.training.optimizer.Optimizer
import ai.djl.util.Ec2Utils
import ai.djl.util.RandomUtils
import ai.djl.util.Utils
import ai.djl.util.cuda.CudaUtils
import org.slf4j.Logger
import org.slf4j.LoggerFactory
import java.io.IOException
import java.nio.file.Files
import java.nio.file.Paths
import java.util.*
import java.util.concurrent.ConcurrentHashMap
import java.util.regex.Pattern
import kotlin.math.min

/**
 * The `Engine` interface is the base of the provided implementation for DJL.
 *
 *
 * Any engine-specific functionality should be provided through this class. In general, it should
 * contain methods to detect information about the usable machine hardware and to create a new
 * [NDManager] and [Model].
 *
 * @see [Engine Guide](http://docs.djl.ai/docs/engine.html)
 *
 * @see EngineProvider
 *
 * @see [The guide on resource
 * and engine caching](http://docs.djl.ai/docs/development/cache_management.html)
 */
abstract class Engine {

    private var defaultDevice: Device? = null

    /**
     * Returns the random seed in DJL Engine.
     *
     * @return seed the seed to be fixed in Engine
     */
    // use object to check if it's set
    var seed: Int? = null
        private set

    /**
     * Returns the alternative `engine` if available.
     *
     * @return the alternative `engine`
     */
    abstract val alternativeEngine: Engine?

    /**
     * Returns the name of the Engine.
     *
     * @return the name of the engine
     */
    abstract val engineName: String

    /**
     * Return the rank of the `Engine`.
     *
     * @return the rank of the engine
     */
    abstract val rank: Int

    /**
     * Returns the version of the deep learning engine.
     *
     * @return the version number of the deep learning engine
     */
    abstract val version: String

    /**
     * Returns whether the engine has the specified capability.
     *
     * @param capability the capability to retrieve
     * @return `true` if the engine has the specified capability
     */
    abstract infix fun hasCapability(capability: String): Boolean

    /**
     * Returns the engine's default [Device].
     *
     * @return the engine's default [Device]
     */
    fun defaultDevice(): Device? {
        if (defaultDevice == null) {
            defaultDevice = when {
                hasCapability(StandardCapabilities.CUDA) && CudaUtils.getGpuCount() > 0 -> Device.gpu()
                else -> Device.cpu()
            }
        }
        return defaultDevice
    }

    val devices: Array<Device?>
        /**
         * Returns an array of devices.
         *
         *
         * If GPUs are available, it will return an array of `Device` of size
         * \(min(numAvailable, maxGpus)\). Else, it will return an array with a single CPU device.
         *
         * @return an array of devices
         */
        get() = getDevices(Int.MAX_VALUE)

    /**
     * Returns an array of devices given the maximum number of GPUs to use.
     *
     *
     * If GPUs are available, it will return an array of `Device` of size
     * \(min(numAvailable, maxGpus)\). Else, it will return an array with a single CPU device.
     *
     * @param maxGpus the max number of GPUs to use. Use 0 for no GPUs.
     * @return an array of devices
     */
    fun getDevices(maxGpus: Int): Array<Device?> {
        var count = gpuCount
        if (maxGpus <= 0 || count <= 0) {
            return arrayOf(Device.cpu())
        }

        count = min(maxGpus.toDouble(), count.toDouble()).toInt()

        val devices = arrayOfNulls<Device>(count)
        for (i in 0 until count) {
            devices[i] = Device.gpu(i)
        }
        return devices
    }

    val gpuCount: Int
        /**
         * Returns the number of GPUs available in the system.
         *
         * @return the number of GPUs available in the system
         */
        get() = when {
            hasCapability(StandardCapabilities.CUDA) -> CudaUtils.getGpuCount()
            else -> 0
        }

    /**
     * Construct an empty SymbolBlock for loading.
     *
     * @param manager the manager to manage parameters
     * @return Empty [SymbolBlock] for static graph
     */
    open fun newSymbolBlock(manager: NDManager?): SymbolBlock? = throw UnsupportedOperationException("Not supported.")

    /**
     * Constructs a new model.
     *
     * @param name the model name
     * @param device the device that the model will be loaded onto
     * @return a new Model instance using the network defined in block
     */
    abstract fun newModel(name: String?, device: Device?): Model?

    /**
     * Creates a new top-level [NDManager].
     *
     *
     * `NDManager` will inherit default [Device].
     *
     * @return a new top-level `NDManager`
     */
    abstract fun newBaseManager(): NDManager

    /**
     * Creates a new top-level [NDManager] with specified [Device].
     *
     * @param device the default [Device]
     * @return a new top-level `NDManager`
     */
    abstract fun newBaseManager(device: Device): NDManager

    /**
     * Returns a new instance of [GradientCollector].
     *
     * @return a new instance of [GradientCollector]
     */
    open fun newGradientCollector(): GradientCollector = throw UnsupportedOperationException("Not supported.")

    /**
     * Returns a new instance of [ParameterServer].
     *
     * @param optimizer the optimizer to update
     * @return a new instance of [ParameterServer]
     */
    open fun newParameterServer(optimizer: Optimizer): ParameterServer = LocalParameterServer(optimizer)

    /**
     * Seeds the random number generator in DJL Engine.
     *
     *
     * This will affect all [Device]s and all operators using Engine's random number
     * generator.
     *
     * @param seed the seed to be fixed in Engine
     */
    open fun setRandomSeed(seed: Int) {
        this.seed = seed
        RandomUtils.RANDOM.setSeed(seed.toLong())
    }

    /** {@inheritDoc}  */
    override fun toString(): String = "$engineName:$version"

    companion object {
        private val logger: Logger = LoggerFactory.getLogger(Engine::class.java)

        private val ALL_ENGINES: MutableMap<String, EngineProvider> = ConcurrentHashMap()

        private val DEFAULT_ENGINE = initEngine()
        private val PATTERN: Pattern = Pattern.compile("KEY|TOKEN|PASSWORD", Pattern.CASE_INSENSITIVE)

        @Synchronized
        private fun initEngine(): String? {
            val loaders = ServiceLoader.load(EngineProvider::class.java)
            for (provider in loaders) {
                registerEngine(provider)
            }

            if (ALL_ENGINES.isEmpty()) {
                logger.debug("No engine found from EngineProvider")
                return null
            }

            val def = System.getProperty("ai.djl.default_engine")
            var defaultEngine = Utils.getenv("DJL_DEFAULT_ENGINE", def)
            if (defaultEngine == null || defaultEngine.isEmpty()) {
                var rank = Int.MAX_VALUE
                for (provider in ALL_ENGINES.values) {
                    if (provider.engineRank < rank) {
                        defaultEngine = provider.engineName
                        rank = provider.engineRank
                    }
                }
            } else if (defaultEngine !in ALL_ENGINES) {
                throw EngineException("Unknown default engine: $defaultEngine")
            }
            logger.debug("Found default engine: {}", defaultEngine)
            Ec2Utils.callHome(defaultEngine)
            return defaultEngine
        }

        @JvmStatic
        val defaultEngineName: String
            /**
             * Returns the default Engine name.
             *
             * @return the default Engine name
             */
            get() = System.getProperty("ai.djl.default_engine", DEFAULT_ENGINE)

        @JvmStatic
        val instance: Engine
            /**
             * Returns the default Engine.
             *
             * @return the instance of `Engine`
             * @see EngineProvider
             */
            get() {
                if (DEFAULT_ENGINE == null) {
                    throw EngineException(
                        ("No deep learning engine found."
                         + System.lineSeparator()
                         + "Please refer to"
                         + " https://github.com/deepjavalibrary/djl/blob/master/docs/development/troubleshooting.md"
                         + " for more details."))
                }
                return getEngine(defaultEngineName)
            }

        /**
         * Returns if the specified engine is available.
         *
         * @param engineName the name of Engine to check
         * @return `true` if the specified engine is available
         * @see EngineProvider
         */
        @JvmStatic
        infix fun hasEngine(engineName: String): Boolean = engineName in ALL_ENGINES

        /**
         * Registers a [EngineProvider] if not registered.
         *
         * @param provider the `EngineProvider` to be registered
         */
        fun registerEngine(provider: EngineProvider) {
            logger.debug("Registering EngineProvider: {}", provider.engineName)
            ALL_ENGINES.putIfAbsent(provider.engineName, provider)
        }

        @JvmStatic
        val allEngines: Set<String>
            /**
             * Returns a set of engine names that are loaded.
             *
             * @return a set of engine names that are loaded
             */
            get() = ALL_ENGINES.keys

        /**
         * Returns the `Engine` with the given name.
         *
         * @param engineName the name of Engine to retrieve
         * @return the instance of `Engine`
         * @see EngineProvider
         */
        @JvmStatic
        fun getEngine(engineName: String): Engine {
            val provider = ALL_ENGINES[engineName]
            requireNotNull(provider) { "Deep learning engine not found: $engineName" }
            return provider.engine
        }

        @JvmStatic
        val djlVersion: String
            /**
             * Returns the DJL API version.
             *
             * @return seed the seed to be fixed in Engine
             */
            get() {
                val version = Engine::class.java.getPackage().specificationVersion
                if (version != null) {
                    return version
                }
                try {
                    Engine::class.java.getResourceAsStream("api.properties").use { `is` ->
                        val prop = Properties()
                        prop.load(`is`)
                        return prop.getProperty("djl_version")
                    }
                } catch (e: IOException) {
                    throw AssertionError("Failed to open api.properties", e)
                }
            }

        /** Prints debug information about the environment for debugging environment issues.  */
        fun debugEnvironment() {
            println("----------- System Properties -----------")
            System.getProperties().forEach { (k, v) -> print(k as String, v) }

            println()
            println("--------- Environment Variables ---------")
            Utils.getenv().forEach { (k, v) -> print(k, v) }

            println()
            println("-------------- Directories --------------")
            try {
                val temp = Paths.get(System.getProperty("java.io.tmpdir"))
                println("temp directory: $temp")
                val tmpFile = Files.createTempFile("test", ".tmp")
                Files.delete(tmpFile)

                val cacheDir = Utils.getCacheDir()
                println("DJL cache directory: " + cacheDir.toAbsolutePath())

                val path = Utils.getEngineCacheDir()
                println("Engine cache directory: " + path.toAbsolutePath())
                Files.createDirectories(path)
                if (!Files.isWritable(path)) {
                    println("Engine cache directory is not writable!!!")
                }
            } catch (e: Throwable) {
                e.printStackTrace(System.out)
            }

            println()
            println("------------------ CUDA -----------------")
            val gpuCount = CudaUtils.getGpuCount()
            println("GPU Count: $gpuCount")
            if (gpuCount > 0) {
                println("CUDA: " + CudaUtils.getCudaVersionString())
                println("ARCH: " + CudaUtils.getComputeCapability(0))
            }
            for (i in 0 until gpuCount) {
                val device = Device.gpu(i)
                val mem = CudaUtils.getGpuMemory(device)
                println("GPU(" + i + ") memory used: " + mem.committed + " bytes")
            }

            println()
            println("----------------- Engines ---------------")
            println("DJL version: $djlVersion")
            println("Default Engine: $instance")
            println("Default Device: " + instance.defaultDevice())
            for (provider in ALL_ENGINES.values) {
                println(provider.engineName + ": " + provider.engineRank)
            }
        }

        private fun print(key: String, value: Any) {
            var v = value
            if (PATTERN.matcher(key).find()) {
                v = "*********"
            }
            println("$key: $v") // NOPMD
        }
    }
}
