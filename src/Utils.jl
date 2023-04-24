module Utils

using PyPlot

# Set the style of the plot. 
PyPlot.plt.style.use("seaborn-v0_8-colorblind")

"""
Plot a graphical representation of the interaction matrix χ. 

# Arguments: 
- `χ::Matrix{Float64}`: The matrix that will be plotted.
"""
function plot_matrix(χ::Matrix{Float64}; 
    title=nothing
)
    N = size(χ, 1)

    # Plot the values. 
    m = maximum(map(abs, χ))

    # Set the labels for the volume fractions. 
    ticks = ["\$Φ_$i\$" for i in 0:N]

    PyPlot.imshow(χ, cmap=:PRGn, clim=(-m,m))
    PyPlot.colorbar()

    ax = PyPlot.gca()

    ax.set_xticklabels(ticks)
    ax.set_yticklabels(ticks)

    if !isnothing(title)
        PyPlot.title(title)
    end
end


"""
Plot a representation of the phases as a pie chart. 

# Arguments: 
- `phases::Vector{Vector{Float64}}`: The vector of phases. This is one of the outputs from `sample_phases`. 
- `title=nothing`: The title of the plot. (If it is nothing, then there will be no title.)
"""
function plot_phases(
    phases::Vector{Vector{Float64}};
    title=nothing, 
    legend=false, 
    subtitles=nothing, 
    component_labels=nothing
)
    N = length(phases[1])
    M = length(phases)
    names = ["\$Φ_$i\$" for i in 1:N]

    fig, ax = PyPlot.subplots(ncols=M)

    for i in 1:M 
        ax[i].pie(phases[i], labels=[" " for i in 1:N])
    end

    if legend 
        handles, labels = PyPlot.gca().get_legend_handles_labels()
        fig.legend(handles[1:N], names, loc="lower center", ncol=N)
    end

    if !isnothing(title)
        fig.suptitle(title, weight="bold")
    end

    if !isnothing(subtitles)
        for i in 1:M 
            ax[i].set_title(subtitles[i])
        end
    end
    
end

"""
Plot a bar chart comparing the volume fraction of didfferent components accross differnet phases 
for different samples. 
"""
function plot_comparitive_phases(
    samples::Vector{Vector{Vector{Float64}}};
    title=nothing, 
    legend=false, 
    subtitles=nothing, 
    component_labels=nothing, 
    component_label_rotation=10, 
    error_bars=nothing
)   
    num_samples = length(samples)
    num_phases = 1 #length(samples[1])
    num_components = length(samples[1][1])
    
    fig, ax = PyPlot.subplots(nrows=num_phases)

    if num_phases == 1 
        ax = [ax]
    end

    component_names = ["Component $i" for i in 1:num_components]
    sample_names = ["Sample $i" for i in 1:num_samples]

    bar_shift = 1.0
    bar_width = bar_shift / (num_samples+1)

    for phase ∈ 1:num_phases

        for sample ∈ 1:num_samples
            component_levels = [samples[sample][phase][component] for component in 1:num_components] 
            pos(x) = x + (sample*bar_width) - ((bar_shift * num_samples / (num_samples+1))/2) - bar_width/2
            ax[phase].bar(map(pos , 1.0:num_components), component_levels, label=sample_names[sample], width=bar_width)

            if !isnothing(error_bars)
                x = map(pos , 1.0:num_components)
                y = component_levels
                y_err = [error_bars[sample][phase][component] for component in 1:num_components] 
                ax[phase].errorbar(x, y, yerr=y_err, capsize=5, ecolor="black", linestyle="")
            end # if !isnothing(error_bars)

        end # for sample ∈ 1:num_samples

        ax[phase].legend()

        ax[phase].set_ylabel("Volume Fraction")

        if !isnothing(component_labels)
            ax[phase].set_xticks(1:bar_shift:bar_shift*length(component_labels), component_labels, rotation=component_label_rotation)
        end
    end # for phase ∈ 1:num_phases

    if !isnothing(title)
        fig.suptitle(title, weight="bold")
    end

    

end # function plot_comparitive_phases

end # module Utils