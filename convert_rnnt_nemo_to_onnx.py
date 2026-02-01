import torch
import argparse
import os
from nemo.collections.asr.models import EncDecRNNTBPEModel

def convert_rnnt_nemo_to_onnx(nemo_path, onnx_path, device='cpu'):
    """
    Convert Parakeet TDT RNNT .nemo model to ONNX for deployment.
    Uses NeMo's built-in export() method for proper ONNX conversion.
    Exports encoder, decoder, and joint networks.
    """
    print(f"Loading RNNT model on {device}...")
    model = EncDecRNNTBPEModel.restore_from(nemo_path)
    model.eval()
    model.to(device)

    # Create output folder based on the output filename
    base_path = onnx_path.rsplit('.', 1)[0]
    output_folder = base_path
    os.makedirs(output_folder, exist_ok=True)
    
    print(f"Output folder: {output_folder}")
    print(f"Exporting ONNX model using NeMo's export method...")
    
    # Try single model export first
    single_output = os.path.join(output_folder, os.path.basename(onnx_path))
    try:
        model.export(
            output=single_output,
            verbose=True,
            do_constant_folding=True,
            onnx_opset_version=17,
            check_trace=False,
        )
        print(f"ONNX model saved successfully at: {single_output}")
        
    except Exception as e:
        print(f"Single model export failed: {e}")
        print("\nExporting RNNT components (encoder/decoder/joint)...\n")
        
        # Export subnets individually - standard for RNNT models
        try:
            subnets = model.list_export_subnets()
            print(f"Found subnets: {subnets}")
            
            essential_files = []
            for subnet_name in subnets:
                subnet = model.get_export_subnet(subnet_name)
                output_file = os.path.join(output_folder, f"{subnet_name}.onnx")
                print(f"Exporting {subnet_name} to {output_file}...")
                subnet.export(
                    output=output_file,
                    verbose=True,
                    do_constant_folding=True,
                    onnx_opset_version=17,
                    check_trace=False,
                )
                essential_files.append(output_file)
                
                # Check for .onnx_data file
                data_file = f"{output_file}_data"
                if os.path.exists(data_file):
                    essential_files.append(data_file)
            
            print(f"\n✓ Successfully exported RNNT model components to: {output_folder}")
            for f in essential_files:
                if os.path.exists(f):
                    size_mb = os.path.getsize(f) / (1024 * 1024)
                    print(f"  - {os.path.basename(f)} ({size_mb:.2f} MB)")
            
            print(f"\nNote: You need all these files for inference:")
            print(f"  1. encoder.onnx (+ encoder.onnx_data if present)")
            print(f"  2. decoder.onnx")
            print(f"  3. joint.onnx (if present)")
            
        except Exception as e2:
            print(f"Subnet export failed: {e2}")
            print("\nThis model may not support ONNX export.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Convert Parakeet RNNT .nemo model to ONNX"
    )
    parser.add_argument("nemo_path", type=str, help="Path to the .nemo model file")
    parser.add_argument("onnx_path", type=str, help="Path to save the .onnx model (creates a folder with this name)")
    parser.add_argument("--device", type=str, default="cpu", help="Device to use: 'cpu' or 'cuda'")
    args = parser.parse_args()

    convert_rnnt_nemo_to_onnx(args.nemo_path, args.onnx_path, device=args.device)
