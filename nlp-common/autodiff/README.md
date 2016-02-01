# autodiff
A very small automatic differentiation library for C++

# Dependencies
You have to install `libeigen3-dev` first. Support is only for ubuntu

# Using it
Take a look at `main.cpp` to read the example. For your own projects, build it,
install it, then link `ad` for your projects.

1. Create a ComputationGraph object.
2. Instantiate the variables. Either from Eigen::MatrixXd that will be copied
   from, or from shared\_ptr to them (like, in parameter). *NOTE:* The Var
   objects MUST NOT outlive the ComputationGraph instance they were created
   from.
3. Do your calculations
4. Backpropagate.

NB: Unlike theano, the ComputationGraph can do only one forward pass. Even if
it creates a lot of overhead for "fixed" size models, it makes sequences
calculation much much easier to code and to backpropagate into. Look at the
example and note how a new graph (which is actually more like a trace of what
happened) is created for every new pass.
