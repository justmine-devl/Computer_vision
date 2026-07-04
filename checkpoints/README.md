# Checkpoints

Do not commit checkpoints or model weights to GitHub.

Place local weights manually when running experiments. Example layout:

```text
checkpoints/
|-- adair/
|   |-- adair-single-dehaze.ckpt
|   |-- adair3d.ckpt
|   `-- adair5d.ckpt
|-- udpnet/
|   `-- udpnet_dehazing.pth
|-- hogformer/
|   `-- best.ckpt
|-- yolo/
|   `-- yolo26n.pt
`-- depth_anything/
    `-- depth_anything_v2_vits.pth
```
