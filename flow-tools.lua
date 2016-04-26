

local flow_node_global_counter = 0

function to_graph(model, out, draw_line)
    draw_line = draw_line or function() end
    -- Label each node
    if not model.flow_node_id then
        model.flow_node_id = "node_"..flow_node_global_counter
        flow_node_global_counter = flow_node_global_counter + 1
    end
    -- draw_line is a function that takes (current_node) and draws
    -- a line from the last node to the current node.
    function make_draw_line(node1)
        return function(node2)
            --print("LINE:",torch.typename(node1), torch.typename(node2))
            out(node1.flow_node_id.." -> "..node2.flow_node_id..";")
        end
    end
    -- Output the graph.
    if torch.typename(model) == 'nn.Sequential' then
        out("subgraph cluster_seq_"..model.flow_node_id.." {")
        out "style = filled;"
        out "color = \"#eeeeff\";"
        out "fillcolor = \"#0000ff04\";"
        for _,module in ipairs(model.modules) do
            local next_mod = to_graph(module, out, draw_line)
            draw_line = next_mod
        end
        out "}"
    elseif torch.typename(model) == 'nn.ConcatTable' then
        out("subgraph cluster_concat_"..model.flow_node_id.." {")
        out "style = filled;"
        out "color = \"#eeeeff\";"
        out "fillcolor = \"#0000ff04\";"
        local more_draw_lines = {}
        for _, module in ipairs(model.modules) do
            more_draw_lines[#more_draw_lines + 1] = to_graph(module, out, draw_line)
            --draw_line(module)
        end
        out "}"
        -- the line goes from each node in the ConcatTable to the next result.
        draw_line = function(next_module)
            for _,draw in ipairs(more_draw_lines) do
                draw(next_module)
            end
        end
    else
        local cleaned_name = torch.typename(model):split("[.]")
        cleaned_name = cleaned_name[#cleaned_name]
        draw_line(model)
        draw_line = make_draw_line(model)
        out(model.flow_node_id.." [label=\""..cleaned_name.."\"];")
    end

    return draw_line
end


require 'cunn' require 'cudnn'
resnet=torch.load('pretrained/resnet-200.t7')
print "digraph {"
print "node [style=filled,fillcolor=white,height=0,shape=box];"
print "graph [ranksep=0.1];"
--resnet2 = resnet.modules[5].modules[1]
--resnet = nn.Sequential()
--resnet:add(nn.View())
--resnet:add(resnet2)
to_graph(resnet, print)
print "}"
--print "---\n"
--print(resnet)
