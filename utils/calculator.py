def calculate_recovery(entry_price, entry_qty, leverage, current_price, target_entry):
    """Menghitung kebutuhan margin recovery"""
    numerator = (entry_price * entry_qty) - (target_entry * entry_qty)
    denominator = target_entry - current_price
    add_qty = numerator / denominator if denominator != 0 else 0

    add_position_value = add_qty * current_price
    add_margin = add_position_value / leverage
    total_qty = entry_qty + add_qty
    new_avg_entry = ((entry_price * entry_qty) + (current_price * add_qty)) / total_qty

    return {
        "add_qty": add_qty,
        "add_position_value": add_position_value,
        "add_margin": add_margin,
        "new_avg_entry": new_avg_entry,
    }
