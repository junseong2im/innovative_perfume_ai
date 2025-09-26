import { CartProvider } from 'components/cart/cart-context';
// import { Navbar } from 'components/layout/navbar';
import GlobalNav from 'components/layout/global-nav';
import { WelcomeToast } from 'components/welcome-toast';
import { GeistSans } from 'geist/font/sans';
import { getCart } from 'lib/shopify';
import { ReactNode } from 'react';
import { Toaster } from 'sonner';
import './globals.css';
import { baseUrl } from 'lib/utils';

const { SITE_NAME } = process.env;

export const metadata = {
  metadataBase: new URL(baseUrl),
  title: {
    default: SITE_NAME!,
    template: `%s | ${SITE_NAME}`
  },
  robots: {
    follow: true,
    index: true
  }
};

export default async function RootLayout({
  children
}: {
  children: ReactNode;
}) {
  // Don't await the fetch, pass the Promise to the context provider
  const cart = getCart();

  return (
    <html lang="en" className={GeistSans.variable}>
      <body className="bg-[var(--luxury-cream)] text-[var(--luxury-midnight)] selection:bg-[var(--luxury-rose-gold)] selection:text-[var(--luxury-cream)] dark:bg-[var(--luxury-obsidian)] dark:text-[var(--luxury-pearl)] dark:selection:bg-[var(--luxury-gold)] dark:selection:text-[var(--luxury-midnight)]">
        <CartProvider cartPromise={cart}>
          {/* <Navbar /> */}
          <GlobalNav />
          <main>
            {children}
            <Toaster closeButton />
            <WelcomeToast />
          </main>
        </CartProvider>
      </body>
    </html>
  );
}
