// ═══════════════════════════════════════════════════════
// CYCLONE DESIGN APP — site.js
// ═══════════════════════════════════════════════════════

document.addEventListener('DOMContentLoaded', function () {

    // ── Auto-dismiss toasts after 4 seconds ────────────────────────
    document.querySelectorAll('.alert-toast').forEach(function (el) {
        setTimeout(function () {
            el.style.transition = 'opacity 0.4s';
            el.style.opacity = '0';
            setTimeout(function () { el.remove(); }, 400);
        }, 4000);
    });

    // ── Collapsible form sections ──────────────────────────────────
    document.querySelectorAll('.form-section-header').forEach(function (header) {
        header.addEventListener('click', function () {
            var targetId = this.dataset.target;
            var target   = document.getElementById(targetId);
            if (!target) return;

            var isCollapsed = target.classList.contains('show');
            target.classList.toggle('show');
            this.classList.toggle('collapsed', isCollapsed);

            var icon = this.querySelector('.toggle-icon');
            if (icon) icon.style.transform = isCollapsed ? 'rotate(-90deg)' : 'rotate(0deg)';
        });
    });

    // ── Confirm dialogs for destructive actions ────────────────────
    document.querySelectorAll('[data-confirm]').forEach(function (el) {
        el.addEventListener('click', function (e) {
            if (!confirm(this.dataset.confirm)) {
                e.preventDefault();
                e.stopPropagation();
            }
        });
    });

    // ── Auto-format numeric inputs on blur ─────────────────────────
    document.querySelectorAll('input[data-format="decimal"]').forEach(function (el) {
        el.addEventListener('blur', function () {
            var val = parseFloat(this.value);
            if (!isNaN(val)) this.value = val.toFixed(parseInt(this.dataset.decimals || '2'));
        });
    });

    // ── Bootstrap tooltips init ────────────────────────────────────
    var tooltipEls = document.querySelectorAll('[data-bs-toggle="tooltip"]');
    tooltipEls.forEach(function (el) {
        new bootstrap.Tooltip(el);
    });

});

// ── Global utility: show inline loading spinner on a button ─────────────────
function setButtonLoading(btn, loading) {
    if (loading) {
        btn.dataset.originalHtml = btn.innerHTML;
        btn.disabled = true;
        btn.innerHTML = '<span class="spinner-border spinner-border-sm me-2"></span>Processing...';
    } else {
        btn.disabled = false;
        btn.innerHTML = btn.dataset.originalHtml || btn.innerHTML;
    }
}
