xdg-open "$(find -follow | grep \\.pdf | fzf --delimiter / --with-nth="3..")"
