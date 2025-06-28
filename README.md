# WGPU with OSC Receiver

A basic setup that instantiates a WGPU instance and a dedicated OSC thread.

WGPU code is a reorganized version of [this](https://github.com/sotrh/learn-wgpu/tree/master/code/beginner/tutorial9-models/), from the [wgpu tutorial](https://sotrh.github.io/learn-wgpu/#what-is-wgpu).

OSC refers to [Open Sound Control](https://ccrma.stanford.edu/groups/osc/index.html). It's used for real-time interfacing with musical instruments. Description from that link: 

"OpenSoundControl (OSC) is a data transport specification (an encoding) for realtime message communication among applications and hardware. OSC was developed by researchers Matt Wright and Adrian Freed during their time at the Center for New Music & Audio Technologies (CNMAT). OSC was originally designed as a highly accurate, low latency, lightweight, and flexible method of communication for use in realtime musical performance. They proposed OSC in 1997 as “a new protocol for communication among computers, sound synthesizers, and other multimedia devices that is optimized for modern networking technology”.

This repo uses the [rosc](https://github.com/klingtnet/rosc) library.