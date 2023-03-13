import numpy as np
import tensor_pb2
import time


def main():
    ser = tensor_pb2.TensorProto.SerializeToString
    x = np.random.randn(8192, 16).astype(np.float32)
    print(x.dtype)

    tensor_proto = tensor_pb2.TensorProto()
    tensor_proto.tensor_content = x.tobytes()
    
    for _ in range(6):
        s = time.time()
        output = ser(tensor_proto)
        e = time.time()
        print(f"Time(ms): {(e-s) * 1000:.2f}")


if __name__ == "__main__":
    main()
    