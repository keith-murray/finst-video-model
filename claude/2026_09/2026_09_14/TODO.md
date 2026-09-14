# Run inference on local gemma with mp4

## Context

We have been attempting to debug a gap in performance in between our locally hosted version of gemma4 and the openrouter hosted versions of gemma4. Offline from our work, I have developed a pipeline for getting `torchcodec` and `ffmpeg` packages to work on the local cluster. Take a look at `claude/skills/cluster/ffmpeg/test_torchcodec_with_module.sh` for context.

## Task 1: Running inference on the pylyshyn mp4s on the cluster

I have rsynced our Pylyshyn mp4s on to the cluster. All we need is a script to run a sweep, aggregate the data, and plot the data. I want the script to be as simple as possible. We shouldn't write any video metadata as the mp4 pipeline should extract everything underneath the hood. In other words, we built a lot of work arounds that we no longer need or want to use.

## Task 2: Testing more trials at n=4

From the previous task, we saw that the mp4 pipeline dramatically helped at n=4 objects. We want to make sure this results holds in the face of even more samples. Let's have 100 samples, 50 true and 50 false. Also, let's make it such that the nearest neighbor heuristic only performs at chance. We can do this by sampling many trials, and then subsampling to get the 100 samples. We also want to test both the numpy and mp4 pipelines.

Generate the data locally via writing a script, and then I will rsync to the cluster. Also write the inference, aggregation, and plotting script. The final plot should be a bar plot with std visualized.

## Task 3: Running the trials from task 2 on openrouter

Just for completion, run the previous trials we generated on openrouter. Choose one host with one condition. Add the results to the plot when done.
