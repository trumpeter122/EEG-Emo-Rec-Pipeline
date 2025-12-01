#!/usr/bin/env nix-shell
#! nix-shell -i zsh -p eza tokei

# tree src -I "__pycache__" -C --dirsfirst --sort=name
#
# echo Total lines:
# echo $(find src -type d -name __pycache__ -prune -o -type f -name '*.py' -print | xargs cat | wc -l)

eza . --tree --git-ignore --group-directories-first
tokei .
