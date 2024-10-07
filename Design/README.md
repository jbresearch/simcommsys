# Design Improvements to Simcommsys

## New Input Language

### Examples

``` json
{
    "Simulator": "commsys_simulator",
    "InputMode": "user",
    "Input": [1,0,1,0,1,0],
    "Codecs": [
        "Name": "codec1",
        "Codec": "ldpc<gf16,float>",
        "Args": {
            "Version": "5",
            "SPAType": "gdl_cuda",
            ...
        },
    ],
    "Mappers": [
        {
            "Name": "mapper1",
            "Mapper": "map_straight<vector,double>",
        }
    ],
    "Modems": [
        {
            "Name": "mymodem1",
            "Modem": "direct_blockmodem<gf16,vector,double>"
        }
    ],
    "Channels": [
        {
            "Name": "mychannel1",
            "Channel": "qsc<gf16>"
        }
    ]
    "System": {
        "ClassName": "commsys",
        "InputType": "vector<gf16>",
        "OutputType": "vector<vector<gf16>>",
        "StartIdx": 0,
        "Nodes": [
            {
                "NodeType": "encode",
                "Codec": "codec1",
                "Outputs": ["encoded"]   
            },
            {
                "NodeType": "map",
                "Mapper": "mapper1",
                "Inputs": ["encoded"],
                "Outputs": ["mapped"]   
            },
            {
                "NodeType": "modulate",
                "Modem": "mymodem1",
                "Inputs": ["mapped"],
                "Outputs": ["modulated"]   
            },
            {
                "NodeType": "transmit",
                "Channel": "mychannel1",
                "Inputs": ["modulated"],
                "Outputs": ["transmitted"]   
            },
            {
                "NodeType": "demodulate",
                "Modem": "mymodem1",
                "Inputs": ["transmitted"],
                "Outputs": ["demodulated"] 
            },
            // ...
            {
                "NodeType": "repeat",
                "OutputType": "vector<gf16>",
                "StartIdx": 0,
                "Nodes": [
                    {
                        "NodeType": "decode",
                        "Codec": "codec1",
                        "Outputs": ["decoded"] 
                    }
                ],
                "NIters": 100,
                "Inputs": ["unmapped"],
                "Outputs": ["decoded"] 
            }
        ]
    }
}
```

### Specifying `commsys_fulliter` using the new language

TODO: Continue

``` json
{
    "Simulator": "commsys_simulator",
    "InputMode": "random",
    "Codecs": [
        "Name": "codec1",
        "Codec": "ldpc<gf16,float>",
        "Args": {
            "Version": "5",
            "SPAType": "gdl_cuda",
            ...
        },
    ],
    "Mappers": [
        {
            "Name": "mapper1",
            "Mapper": "map_straight<vector,double>",
        }
    ],
    "Modems": [
        {
            "Name": "mymodem1",
            "Modem": "direct_blockmodem<gf16,vector,double>"
        }
    ],
    "Channels": [
        {
            "Name": "mychannel1",
            "Channel": "qsc<gf16>"
        }
    ]
    "System": {
        "ClassName": "commsys",
        "InputType": "vector<gf16>",
        "OutputType": "vector<vector<gf16>>",
        "StartIdx": 0,
        "Nodes": [
            {
                "NodeType": "encode",
                "Codec": "codec1",
                "Outputs": ["encoded"]   
            },
            {
                "NodeType": "map",
                "Codec": "mapper1",
                "Inputs": ["encoded"],
                "Outputs": ["mapped"]   
            },
            {
                "NodeType": "modulate",
                "Codec": "mymodem1",
                "Inputs": ["mapped"],
                "Outputs": ["modulated"]   
            },
            {
                "NodeType": "transmit",
                "Codec": "mychannel1",
                "Inputs": ["modulated"],
                "Outputs": ["transmitted"]   
            },
            {
                "NodeType": "zeros",
                "Type": "vector<vector<double>>",
                "Outputs": ["ptable_ext_modem"] 
            },
            {
                "NodeType": "repeat",

            }
            // ... CONTINUE
            {
                "NodeType": "informed_demodulate",
                "Modem": "stream_modulator<gf16, vector>",
                "Inputs": []
            }
        ]
    }
}
```

### Specifying `commsys_stream` using the new language

TODO: Continue

## Implementation

### Node types

A `node` represents a stage in the communication system, e.g. encoding, decoding, transmission, etc.

The entire `commsys` is represented by a DAG of `node` objects (see below).

Nodes can have multiple inputs but always have one output.

The proposed interface for `node` is
``` cpp
class node {
private:
    std::string output_name;
    libbase::vector<std::string> input_names;

public:
    virtual std::any run(const std::unordered_map<std::string, std::any> &inputs) = 0;
    virtual libbase::vector<std::type_info &> get_input_types() const = 0;
    virtual std::type_info &get_output_type() const = 0;

    const libbase::vector<std::string &> get_input_names() const { 
        return this->input_names;
    }

    const std::string &get_output_name() const { 
        return this->output_name;
    }

    virtual const std::string &get_name() const = 0;
};
```

Each different stage of a communication system (e.g. encoding, transmission, etc.) must then implement `node`, e.g. the "encode" stage could be implemented as:
``` cpp
template <class Codec, class C, class dbl>
class encode_node : public node {
private:
    Codec<C, dbl> cdc;

public:
    std::any run(const std::unordered_map<std::string, std::any> &inputs) override {
        // check that only one input is specified for encode.
        assertalways(this->get_input_names().size() == 1);

        try {
            C<dbl> encoded;
            // get the input
            C<dbl> *source = std::any_cast<C<dbl>>(&inputs[this->get_input_names()(0)]);
            cdc.do_encode(*source, encoded);
            return std::any(std::move(encoded));
        } catch (std::bad_any_cast &) {
            failwith("Invalid input type given.");
        }
    }
    libbase::vector<std::type_info &> get_input_types() const override {
        return {typeid(C<dbl>)};
    }
    std::type_info &get_output_type() const override {
        return typeid(C<dbl>);
    }

    const std::string &get_name() const override {
        return "encode";
    }
};
```

### DAG Type

TODO: Extend DAG to work with multiple inputs.

``` cpp
class dag {
private:
    // list of topologically sorted nodes.
    std::list<std::unique_ptr<node>> nodes;
    // map each input name to the nodes which have it as their input.
    std::unordered_map<std::string, libbase::vector<node *>> inputs;
    node *start, *exit;

    // store outputs of each node.
    std::unordered_map<std::string, std::any> outputs;

    dag(std::list<unique_ptr<node>> &&nodes, 
        std::unordered_map<std::string, libbase::vector<node *>> &&inputs,
        int startindex = 0)
        : nodes(nodes), inputs(inputs), start(&(this->nodes[startindex])) {
            // 1. topologically sort list of nodes

            // 2. find exit node and set this->exit to point to it.

            // 3. validate DAG
        }

    std::type_info &get_input_type() const {
        return start->get_input_types()(0);
    }
    std::type_info &get_output_type() const {
        return exit->get_output_type();
    }

    std::any run(const std::any &input) {
        outputs[start.get_output_name()] = start.run(input);
        for (auto it = ++nodes.begin(); it != nodes.end(); ++it)
            outputs[it->get_output_name()] = it->run(outputs);
        return outputs[exit.get_output_name()];
    }
};
```

The DAG needs to perform a number of validating checks in its constructor; namely that
- The set of nodes with their inputs and outputs actually forms a DAG, and can be topologically sorted
- `start.get_input_names().size() == 1`, i.e. the starting node takes only one input.
- There is exactly one `exit` node, i.e. any path starting from the `start` node and following the DAG will converge onto a single node.
- Check that where there is a connection between two nodes, the output type of the source (indicated by `node::get_output_type()`) matches the input type of the sink (indicated by `node::get_input_type()`).

### Repeat nodes

In certain cases we want to apply a part of the communication system pipeline for a fixed number of iterations, collecting the results from each run. This is accomplished by a special type of node called a `repeat` node, which can be implemented as:
``` cpp
template <class OutputType>
class repeat_node : public node {
private:
    dag d;
    int num_iters;

public:
    std::any run(const std::unordered_map<std::string, std::any> &inputs) override {
        // check that only one input is specified for encode.
        assertalways(this->get_input_names().size() == 1);

        libbase::vector<OutType> results(num_iters);
        // run through nodes in the DAG num_iters times.
        results(0) = d.run(
            inputs[this->get_input_names()(0)]
        );
        for (int i = 1; i < num_iters; i++) {
            std::any res = d.run(results(i-1));
            try {
                // store result in results vector.
                results(i) = std::any_cast<OutputType>(res);
            } catch (std::bad_any_cast &) {
                failwith("Invalid output type given.");
            }
        }
        return results;
    }
    libbase::vector<std::type_info &> get_input_types() const override {
        return {typeid(OutputType)};
    }
    std::type_info &get_output_type() const override {
        // Note that output type is vector of outputs from codec,
        // as we return results from each iteration.
        return typeid(libbase::vector<OutputType>);
    }

    const std::string &get_name() const override {
        return "repeat";
    }
};
```

The `repeat` node needs to perform a number of validating checks in its constructor; namely that
- `d.get_output_type() == d.get_input_type()` so that output can be fed back into input of the DAG between iterations.
- `d.get_output_type() == typeid(OutputType)` so that we know that DAG computes the right kind of result.

### The `commsys` class

The proposed interface for the new `commsys` class is:
``` cpp
template <class InputType, class OutputType>
class commsys_inf {
public:
    virtual OutputType fullcycle(const InputType &source) = 0;
    // ...
};
```

Internally the `commsys` class will have the following members:
``` cpp
template <class InputType, class OutputType>
class commsys : public commsys_inf<InputType, OutputType> {
private:
    dag d;

public:
    OutputType fullcycle(const InputType &source) override;
};
```
with the `run()` method implemented as:
``` cpp
template <class InputType, class OutputType>
OutputType
commsys<InputType, OutputType>::fullcycle(const InputType &source) {
    std::any received = d.run(source)
    // cast result and return
    try {
        return any_cast<OutputType>(std::move(received.res));
    } catch (std::bad_any_cast &) {
        failwith("Invalid output type given.");
    }
}
```

The `commsys` class should also perform the following validation checks in its constructor:
- `d.get_input_type() == typeid(InputType)` so that we know the DAG takes in the right kind of input.
- `d.get_output_type() == typeid(OutputType)` so that we know that DAG computes the right kind of result.

This new `commsys` template class needs to be instantiated for the following types:
``` cpp
/* Serialization string: 
    commsys<container<type>, container<type>> |
    commsys<container<type>, vector<container<type>>>
 * where:
 *      type = sigspace | bool | gf2 | gf4 ...
 *      container = vector | matrix
 */
```

### Changes to `experiment` and `resultscollector` classes

### Serialization