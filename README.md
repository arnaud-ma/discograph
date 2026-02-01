# Discograph

Visualize your Discord friends network as an interactive graph where each node represents a friend and edges represent mutual friends.

## Very small summary
1. Grab your Discord user token from [this link](https://gist.github.com/MarvNC/e601f3603df22f36ebd3102c501116c6) (the first method at the top, not the AI-generated garbage in the comments below).

2. Open a terminal

3. Install `uv` if not already installed:

	Windows:
	```
	powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
	```

   macOS/Linux:
	```
	curl -LsSf https://astral.sh/uv/install.sh | sh
	```

	Then restart your terminal (quit and reopen it).

4. Run the app:
   ```
   uvx discograph
   ```
   And paste your user token if asked.

5. Once the graph is generated and opened, click on the `enabled` button on the `Physics` section.

## Important security warning

> [!CAUTION]
> This tool uses your Discord user token since it's the only way to access your common friends for any give friend.
>
> 1. NEVER, under any circumstance, share your user token with anyone as it would give them full access to your account. It also means you are trusting this tool or have reviewed its code to ensure it is safe.
> 2. Using this is against Discord's Terms of Service. Use at your own risk. For a single personal use, I don't think there is much risk, but be aware of it.

See [this link](https://gist.github.com/MarvNC/e601f3603df22f36ebd3102c501116c6) to get your user token. The first method at the top, not the AI-generated garbage in the comments below.

> [!NOTE]
> Once on the page, click on the `enabled` button on the `Physics` section to make the magic happen!


## Features

- Fetches your Discord friends and their mutual friends.
- Constructs a network graph where nodes represent friends and edges represent mutual friendships.
- Outputs the graph as an interactive HTML file.


## Installation

### Recommended: using uv

1. Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you don't have it already.

	**Windows**:
	```powershell
	powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
	```

	**macOS/Linux**:
	```bash$$
	curl -LsSf https://astral.sh/uv/install.sh | sh
	```

2. Then, either:
	- install the tool
	```bash
	uv tool install discograph
	```
	- or run it without installing everything, letting uv handle everything for you, by prefixing every following commands with `uvx`

### Anything else

It's just a normal Python package, so you can use any other tool that installs python packages.

## Usage

### TLDR;

The most basic and common usage is just the bare command:

```bash
discograph
```

it will:
- ask for your Discord user token
- download your friends and their mutual friends if not already downloaded
- generate the graph in a cache, and open it in your default browser

To refresh the cache (i.e. always download the data from Discord even if already present), use the `--update` flag:

```bash
discograph --update
```

Instead of interactively inputting your user token, you can provide it by:
- setting the `DISCORD_USER_TOKEN` environment variable
- passing it with the `--user-token` option


### Advanced

The default command is in fact a combination of two commands: `download` and `generate-graph`. It is equivalent to:

```bash
discograph download --path mutual_friends.json
discograph generate-graph --input mutual_friends.json --output graph.html
```
but using the default cache paths to store the data.

All commands supports reading/writing from/to stdin/stdout by using `-` as path. It allows for easy piping between commands, e.g. those three commands are equivalent:

```bash
discograph --update
```
```bash
discograph download - | discograph generate-graph -
```
```bash
discograph download mutual_friends.json
cat mutual_friends.json | discograph - -
```

Except that the first one uses the cache paths, while the two others use the current directory and stdin/stdout.

## Documentation

Output help messages for all commands:

```bash
$ discograph --help
Usage: discograph COMMAND [OPTIONS] [ARGS]

Save your Discord mutual friends and visualize them as a graph.

The default behavior is to download the mutual friends data (if not already present) and generate the HTML graph,
then open it in the default web browser.

╭─ Commands ────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ clear        Clear the cached data files.                                                                         │
│ download     Download the mutual friends data from the Discord API and save it to JSON.                           │
│ graph        Create the friends network graph HTML from the mutual friends JSON data.                             │
│ --help (-h)  Display this message and exit.                                                                       │
│ --version    Display application version.                                                                         │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
╭─ Parameters ──────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ DATA-PATH --data-path --input -i       Where to read or save the mutual friends JSON data. Default is in the      │
│                                        operating system's cache directory. Can use "-" for stdin. Does not        │
│                                        supports writing to stdout. Use the download command instead for that.     │
│ HTML-PATH --html-path --output -o      Where to save the generated HTML graph. Default is in the operating        │
│                                        system's cache directory. Can use "-" for stdout.                          │
│ --user-secret                          Set the user secret to use for fetching data. Takes precedence over the    │
│                                        shell variable.                                                            │
│ --redownload --update --no-redownload  Redownload the data from the Discord API, even if it already exists.       │
│   --no-update                          [default: False]                                                           │
│ --progress --no-progress               Show progress bars during data fetching. [default: True]                   │
│ --loglevel                        [choices: DEBUG, INFO, WARNING, ERROR, CRITICAL] [default: WARNING]        │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

```
$ discograph download --help
Usage: default-command download [OPTIONS] [ARGS]

Download the mutual friends data from the Discord API and save it to JSON.

╭─ Parameters ──────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ PATH --path -p             The path to save the mutual friends JSON data. Default is in the operating system's    │
│                            cache directory. Can use "-" for stdout.                                               │
│ USER-SECRET --user-secret                                                                                         │
│ --progress --no-progress   Whether to show progress bars during data fetching. [default: True]                    │
│ --loglevel            [choices: DEBUG, INFO, WARNING, ERROR, CRITICAL] [default: WARNING]                    │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

```
$ discograph graph --help
Usage: default-command graph [ARGS]

Create the friends network graph HTML from the mutual friends JSON data.

╭─ Parameters ──────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ INPUT --input -i                                                                                                  │
│ OUTPUT --output -o             The path to save the generated HTML graph. Default is in the operating system's    │
│                                cache directory. Can use "-" for stdout.                                           │
│ loglevel --loglevel  [choices: DEBUG, INFO, WARNING, ERROR, CRITICAL] [default: WARNING]                │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

```
$ discograph clear --help
Usage: default-command clear [OPTIONS]

Clear the cached data files.

╭─ Parameters ──────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ --all-version --no-all-version  Clear everything, including files from previous versions. [default: False]        │
╰───────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

## Possible improvements

- Currently, some values are hardcoded and are bad values for e.g. small networks:
    - node size
    - edge transparency
    - font size

  A "just work" solution would be to inject some buttons/sliders in the interactive graph to tweak those values on the fly.
  Also, maybe adjust those values based on the number of nodes/edges/clusters detected.

- Add a way to export the graph to other formats:
    - static image (png, jpg, svg, etc.)
	- graph formats (gexf, graphml, etc.) for further analysis in dedicated tools

  Some of those formats would require downloading the avatars instead of hotlinking them in
  the HTML.