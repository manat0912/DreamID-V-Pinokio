module.exports = {
    run: [
        {
            "method": "log",
            "params": {
                "text": "Reassembling split model files..."
            }
        },
        {
            "when": "{{platform === 'win32'}}",
            "method": "shell.run",
            "params": {
                "message": [
                    "copy /b diffusion_pytorch_model.safetensors.part* diffusion_pytorch_model.safetensors",
                    "copy /b models_t5_umt5-xxl-enc-bf16.pth.part* models_t5_umt5-xxl-enc-bf16.pth"
                ],
                "path": "app/models/Wan/Wan2.1-T2V-1.3B"
            }
        },
        {
            "when": "{{platform === 'win32'}}",
            "method": "shell.run",
            "params": {
                "message": [
                    "copy /b dreamidv.pth.part* dreamidv.pth",
                    "copy /b dreamidv_faster.pth.part* dreamidv_faster.pth"
                ],
                "path": "app/models/DreamID-V"
            }
        },
        {
            "when": "{{platform !== 'win32'}}",
            "method": "shell.run",
            "params": {
                "message": [
                    "cat diffusion_pytorch_model.safetensors.part* > diffusion_pytorch_model.safetensors",
                    "cat models_t5_umt5-xxl-enc-bf16.pth.part* > models_t5_umt5-xxl-enc-bf16.pth"
                ],
                "path": "app/models/Wan/Wan2.1-T2V-1.3B"
            }
        },
        {
            "when": "{{platform !== 'win32'}}",
            "method": "shell.run",
            "params": {
                "message": [
                    "cat dreamidv.pth.part* > dreamidv.pth",
                    "cat dreamidv_faster.pth.part* > dreamidv_faster.pth"
                ],
                "path": "app/models/DreamID-V"
            }
        }
    ]
}
