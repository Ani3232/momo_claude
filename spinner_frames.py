"""
# Braille spinner (smooth continuous)
frames = itertools.cycle(["⠁", "⠃", "⠇", "⠏", "⠟", "⠿", "⠾", "⠼", "⠸", "⠰", "⠠", "⡀", "⣀", "⣄", "⣤", "⣦", "⣶", "⣷", "⣿"])

# Simple 4-frame spinner
frames = itertools.cycle(["|", "/", "-", "\\"])

# Arrow spinner
frames = itertools.cycle(["←", "↖", "↑", "↗", "→", "↘", "↓", "↙"])

# Bouncing ball
frames = itertools.cycle(["( ●    )", "(  ●   )", "(   ●  )", "(    ● )", "(     ●)", "(    ● )", "(   ●  )", "(  ●   )"])

# Clock spinner
frames = itertools.cycle(["🕐", "🕑", "🕒", "🕓", "🕔", "🕕", "🕖", "🕗", "🕘", "🕙", "🕚", "🕛"])
    
# Dots expanding
frames = itertools.cycle([" ", ".", "..", "...", "....", ".....", "....", "...", "..", "."])

# Moon phases
frames = itertools.cycle(["🌑", "🌒", "🌓", "🌔", "🌕", "🌖", "🌗", "🌘"])

# Weather based
frames = itertools.cycle(["☀️", "⛅", "☁️", "🌧️", "⛈️", "🌤️"])

# Geometric shapes
frames = itertools.cycle(["◐", "◓", "◑", "◒"])

# Playing card suits
frames = itertools.cycle(["♠", "♣", "♥", "♦"])

# Chess pieces
frames = itertools.cycle(["♙", "♘", "♗", "♖", "♕", "♔"])

# Eye blink
frames = itertools.cycle(["◉", "◌", "◉", "◌"])

# Random
frames = itertools.cycle(["○", "●", "◙", "◦", "╸", "☼"])

"""
# ======================================================================||
#                             SPINNER                                   ||
# ======================================================================||
THINKING_VERBS = [
    "Reading files...               ",
    "Writing code...                ",
    "Running tests...               ",
    "Debugging issues...            ",
    "Analyzing logs...              ",
    "Checking syntax...             ",
    "Installing packages...         ",
    "Committing changes...          ",
    "Searching codebase...          ",
    "Refactoring functions...       ",
    "Fixing bugs...                 ",
    "Building project...            ",
    "Running linter...              ",
    "Formatting code...             ",
    "Creating files...              ",
    "Deleting cruft...              ",
    "Merging branches...            ",
    "Resolving conflicts...         ",
    "Optimizing performance...      ",
    "Adding comments...             ",
    "Removing console.log()...      ",
    "Updating dependencies...       ",
    "Generating docs...             ",
    "Compiling TypeScript...        ",
    "Bundling assets...             ",
    "Deploying to prod...           ",
    "Rolling back...                ",
    "Restoring from backup...       ",
    "Profiling memory...            ",
    "Tracing execution...           ",
]
