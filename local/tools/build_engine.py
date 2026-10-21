import tensorrt as trt
import os
import sys

def build_engine(onnx_path, engine_path, batch_size=1024):
    # 1. Setup Logger
    TRT_LOGGER = trt.Logger(trt.Logger.INFO)
    
    # 2. Create Builder, Network, and Config
    builder = trt.Builder(TRT_LOGGER)
    
    # TRT 11+: EXPLICIT_BATCH is the default and the flag is removed
    try:
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    except AttributeError:
        network = builder.create_network()
    config = builder.create_builder_config()
    
    # 3. Parser
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    print(f"Parsing ONNX model from {onnx_path}...")
    # Use parse_from_file instead of parse(read()) so it can find external data files
    if not parser.parse_from_file(onnx_path):
        print('ERROR: Failed to parse the ONNX file.')
        for error in range(parser.num_errors):
            print(parser.get_error(error))
        return False
            
    print("ONNX parsed successfully.")
    
    # 4. Optimization Profile (for dynamic batch size)
    # Even if we want fixed batch size, it's good practice to define profile
    profile = builder.create_optimization_profile()
    
    # Iterate through all inputs and set profiles
    num_inputs = network.num_inputs
    for i in range(num_inputs):
        tensor = network.get_input(i)
        name = tensor.name
        shape = tensor.shape
        print(f"Configuring Input: {name}, Shape: {shape}")
        
        # 'input': (Batch, 17, 19, 19)
        if name == 'input':
             # Min=1, Opt=64, Max=1024
             profile.set_shape(name, (1, 17, 19, 19), (64, 17, 19, 19), (batch_size, 17, 19, 19))
             
        # 'move1_idx': (Batch,)
        elif name == 'move1_idx':
             profile.set_shape(name, (1,), (64,), (batch_size,))
             
    config.add_optimization_profile(profile)
    
    # 5. Precision: TRT 11+ removed FP16 flag, uses TF32 + default mixed precision
    try:
        config.set_flag(trt.BuilderFlag.FP16)
        print("Enabling FP16 mode...")
    except AttributeError:
        print("TRT 11+: Using default mixed precision (FP16 not a separate flag)")
        
    # Workspace size
    # set_memory_pool_limit is the new API
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4096 * 1024 * 1024) # 4GB
    
    # 6. Build Engine
    print("Building TensorRT engine... This may take a few minutes.")
    serialized_engine = builder.build_serialized_network(network, config)
    
    if serialized_engine is None:
        print("Failed to build engine.")
        return False
        
    print(f"Saving engine to {engine_path}...")
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
        
    print("Engine built and saved successfully!")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python build_engine.py <onnx_path> <engine_path>")
        sys.exit(1)
        
    onnx_p = sys.argv[1]
    engine_p = sys.argv[2]
    
    try:
        import tensorrt
        print(f"TensorRT Version: {tensorrt.__version__}")
        build_engine(onnx_p, engine_p, batch_size=1024)
    except ImportError:
        print("TensorRT python library not found. Please install it.")
