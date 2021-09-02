EMACS = emacs -q --batch --load config.el
define tangle
$(EMACS) $(1) --eval '(org-babel-tangle)'
endef

