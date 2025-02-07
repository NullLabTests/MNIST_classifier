using Pkg
Pkg.add(["Flux", "Images", "MLDatasets", "BSON"])

using Flux, MLDatasets, BSON

# Load MNIST dataset
train_x, train_y = MNIST.traindata()
test_x, test_y = MNIST.testdata()

# Preprocess data
train_x = reshape(train_x, 28*28, :) ./ 255  # Normalize pixel values
test_x = reshape(test_x, 28*28, :) ./ 255

# Define the model
model = Chain(
    Dense(28*28, 32, relu),
    Dense(32, 10),
    softmax)

# Loss function
loss(x, y) = Flux.crossentropy(model(x), Flux.onehotbatch(y, 0:9))

# Optimizer
opt = ADAM()

# Training loop
println("Training...")
for epoch in 1:10
    Flux.train!(loss, params(model), [(train_x[:, i], train_y[i]) for i in 1:60000], opt)
    println("Epoch $epoch")
end

# Evaluate model
accuracy(x, y) = mean(Flux.onecold(model(x)) .== y)
test_accuracy = accuracy(test_x, test_y)
println("Test accuracy: ", test_accuracy)

# Save model
@save "mnist_classifier.bson" model

# Create a README.md file
open("README.md", "w") do f
    write(f, """
    # MNIST Classifier

    This project implements a simple neural network for classifying handwritten digits from the MNIST dataset using Julia and Flux.jl.

    ## Installation
    ```julia
    using Pkg
    Pkg.add(["Flux", "MLDatasets", "BSON"])
    ```

    ## Running the Project
    Clone this repository, navigate to the directory, and run:
    ```julia
    include("mnist_classifier.jl")
    ```

    ## Results
    - **Test Accuracy:** $test_accuracy
    """)
end

# Create a .gitignore file
write(".gitignore", "data/\n*.bson\n")

# Initialize git repository and commit
run(`git init`)
run(`git add .`)
run(`git commit -m "Initial commit with MNIST classifier"`)

# Create a new GitHub repository and push
println("Now create a new GitHub repository named 'MNIST_classifier' and run:")
println("gh repo create MNIST_classifier --public --source=./ --push")
