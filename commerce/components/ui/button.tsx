import { ButtonHTMLAttributes, forwardRef } from 'react';
import clsx from 'clsx';

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'ghost';
  size?: 'sm' | 'md' | 'lg';
}

const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className, variant = 'primary', size = 'md', children, ...props }, ref) => {
    return (
      <button
        ref={ref}
        className={clsx(
          'inline-flex items-center justify-center font-medium transition-all duration-300',
          'focus:outline-none focus:ring-2 focus:ring-[var(--luxury-gold)] focus:ring-offset-2',
          'disabled:opacity-50 disabled:cursor-not-allowed',
          {
            // Variants
            'bg-[var(--luxury-midnight)] text-[var(--luxury-cream)] hover:bg-[var(--luxury-charcoal)] active:bg-[var(--luxury-obsidian)]':
              variant === 'primary',
            'border-2 border-[var(--luxury-midnight)] bg-transparent text-[var(--luxury-midnight)] hover:bg-[var(--luxury-pearl)] active:bg-[var(--luxury-silk)]':
              variant === 'secondary',
            'bg-transparent text-[var(--luxury-midnight)] hover:bg-[var(--luxury-pearl)] active:bg-[var(--luxury-silk)]':
              variant === 'ghost',
            // Sizes
            'px-3 py-1.5 text-sm rounded-md': size === 'sm',
            'px-5 py-2.5 text-base rounded-lg': size === 'md',
            'px-7 py-3.5 text-lg rounded-xl': size === 'lg',
          },
          className
        )}
        {...props}
      >
        {children}
      </button>
    );
  }
);

Button.displayName = 'Button';

export default Button;