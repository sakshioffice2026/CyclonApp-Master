using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using CyclonApp.Database;
using Microsoft.AspNetCore.Http;

namespace CyclonApp.Repositories.Repositories
{
    public class ExceptionHandlerRepository
    {
        private readonly ApplicationDbContext _DbContext;

        public ExceptionHandlerRepository(ApplicationDbContext context)
        {
            this._DbContext = context;
        }

        public void SaveException(string Classname, string methodName, string Error)
        {
            exceptionhandler obj = new exceptionhandler();
            obj.classname = Classname;
            obj.methodname = methodName;
            obj.error = Error;
            obj.datetime = DateTime.Now;
            obj.userid = 0;
            this._DbContext.exceptionhandler.Add(obj);
            try
            {
                this._DbContext.SaveChanges();
            }
            catch (Exception ex)
            {
                // Handle any exceptions that occur during save
                // You might want to log this exception or take other actions
                Console.WriteLine($"Error saving exception: {ex.Message}");
            }
        }
    }
}