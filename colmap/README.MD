# Running the Container with Podman

Please refer to the [documentation](https://gateway.insait.ai/doc/containers.html) for detailed instructions on running containers.

## Building the Container

To build the container image, use the following command:

```bash
podman build -t colmap-cuda:latest .
```

## Running the Container

To run the container with `slurm-podman-run`, execute the following command:

```bash
slurm-podman-run --rm -it \
    -v $WorkFolder:/data \ # Mount the local machine folder to /data as specified in the Dockerfile
    colmap-cuda:latest colmap automatic_reconstructor \
    --workspace_path /data/$WorkFolder \
    --image_path /data/$WorkFolder/images
```

### Notes

1. **Folder Mounting**  
   Replace `$WorkFolder` with the path to your desired working directory on the local machine. This folder will be mounted to `/data` inside the container, as defined in the Dockerfile.

2. **Command Explanation**  
   - `--rm`: Automatically remove the container when it exits.  
   - `-it`: Run the container in interactive mode.  
   - `-v $WorkFolder:/data`: Mount the specified local folder to `/data` in the container.  
   - `colmap-cuda:latest`: Use the `colmap-cuda` image tagged as `latest`.  
   - `colmap automatic_reconstructor`: Invoke the COLMAP automatic reconstructor with the specified paths.  

3. **Workspace and Images Path**  
   The `--workspace_path` and `--image_path` options must point to the corresponding directories within the mounted `/data` folder.
